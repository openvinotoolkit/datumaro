# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import os.path as osp

import h5py
import numpy as np
import json
import cv2
import pandas as pd

from datumaro.components.annotation import AnnotationType, DepthAnnotation, LabelCategories, Points, Skeleton
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.media import Image


class SleapExtractor(SourceExtractor):
    def __init__(self, path, subset=None):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        super().__init__(subset=subset)

        self._categories = self._load_categories(path)
        self._items = list(self._load_items(path).values())

    @staticmethod
    def _read_skeleton(labels_path):
        with h5py.File(labels_path, "r") as f:
            attrs = dict(f["metadata"].attrs)
        metadata = json.loads(attrs["json"].decode())

        # Get node names. This is a superset of all nodes across all skeletons. Note that
        # node ordering is specific to each skeleton, so we'll need to fix this afterwards.
        node_names = [x["name"] for x in metadata["nodes"]]

        # TODO: Support multi-skeleton?
        skeleton = metadata["skeletons"][0]

        # Parse out the cattr-based serialization stuff from the skeleton links.
        edge_inds = []
        for link in skeleton["links"]:
            if "py/reduce" in link["type"]:
                edge_type = link["type"]["py/reduce"][1]["py/tuple"][0]
            else:
                edge_type = link["type"]["py/id"]

            if edge_type == 1:  # 1 -> real edge, 2 -> symmetry edge
                edge_inds.append((link["source"], link["target"]))

        # Re-index correctly.
        skeleton_node_inds = [node["id"] for node in skeleton["nodes"]]
        node_names = [node_names[i] for i in skeleton_node_inds]
        edge_inds = [(skeleton_node_inds.index(s), skeleton_node_inds.index(d)) for s, d in edge_inds]

        return node_names, edge_inds

    def _load_categories(self, path):
        label_cat = LabelCategories()
        node_names, edge_inds = self._read_skeleton(path)
        elements = []
        for i, node_name in enumerate(node_names):
            elements.append({ 'label': node_name, 'element_id': i + 1 })

        edges = []
        for edge in edge_inds:
            edges.append({ 'from': edge[0] + 1, 'to': edge[1] + 1 })

        label_cat.add('skeleton', elements=elements, edges=edges, type='skeleton')

        for node_name in node_names:
            label_cat.add(node_name, parent='skeleton', type='points')

        return {AnnotationType.label: label_cat}

    @staticmethod
    def _read_hdf5(filename, dataset="/"):
        data = {}
        def read_datasets(k, v):
            if type(v) == h5py.Dataset:
                data[v.name] = v[()]

        with h5py.File(filename, "r") as f:
            if type(f[dataset]) == h5py.Group:
                f.visititems(read_datasets)
            elif type(f[dataset]) == h5py.Dataset:
                data = f[dataset][()]
        return data

    def _get_frame_instances(self, labels_path, video, frame_idx):

        # Load metadata.
        frames = self._read_hdf5(labels_path, "frames")
        frames = pd.DataFrame(frames)
        frames.set_index("frame_id", inplace=True)
        instances = self._read_hdf5(labels_path, "instances")
        instances = pd.DataFrame(instances)
        instances.set_index("instance_id", inplace=True)
        points = self._read_hdf5(labels_path, "points")
        points = pd.DataFrame(points)
        pred_points = self._read_hdf5(labels_path, "pred_points")
        pred_points = pd.DataFrame(pred_points)
        tracks = [json.loads(x) for x in self._read_hdf5(labels_path, "tracks_json")]
        tracks = pd.DataFrame(tracks, columns=["start_frame", "track_name"])
        tracks.rename_axis("track", inplace=True)

        # Look up the frame by video and frame_idx.
        frame = frames.reset_index().set_index(["video", "frame_idx"], drop=False).loc[(video, frame_idx)]

        # Get instance ids for the frame.
        instance_ids = np.arange(frame.instance_id_start, frame.instance_id_end)

        # Get instances from ids.
        frame_instances = instances.loc[instance_ids]

        # Parse instances to get points and metadata.
        frame_points = []
        frame_is_predicted = []
        frame_scores = []
        frame_track = []
        for _, instance in frame_instances.iterrows():

            # Get point ids.
            point_ids = np.arange(instance.point_id_start, instance.point_id_end)

            # Save instance-level metadata.
            frame_scores.append(instance.score)
            if instance.track != -1:  # TODO: Check what the actual value is when no track is specified. Might be -1 or NaN?
                frame_track.append(tracks.loc[instance.track].track_name)
            else:
                frame_track.append(None)

            # Get points from the right table based on instance type.
            if instance.instance_type == 0:  # user instance
                frame_is_predicted.append(False)
                instance_points = points.loc[point_ids]
            else:  # predicted instance
                frame_is_predicted.append(True)
                instance_points = pred_points.loc[point_ids]

            # Build (nodes, 2) array.
            pts = np.array([instance_points.x, instance_points.y]).T
            pts[~instance_points.visible] = np.nan
            frame_points.append(pts)

        return frame_points, frame_is_predicted, frame_scores, frame_track

    def _load_items(self, path):
        items = {}

        node_names, _ = self._read_skeleton(path)
        videos = [json.loads(x) for x in self._read_hdf5(path, "videos_json")]
        for video_id in range(len(videos)):
            dset = videos[video_id]["backend"]["dataset"]

            frame_numbers = self._read_hdf5(path, dset.replace("/video", "/frame_numbers")).tolist()
            for frame_id in frame_numbers:
                image_bytes = self._read_hdf5(path, dset)
                frame_ind = frame_numbers.index(frame_id)
                image = cv2.imdecode(image_bytes[frame_ind], cv2.IMREAD_UNCHANGED)

                annotations = []
                frame_points, _, _, _ = self._get_frame_instances(path, video_id, frame_id)
                for points in frame_points:
                    elements = []
                    for point, node_name in zip(points, node_names):
                        elements.append(Points(point, label=self._categories[AnnotationType.label].find(node_name)[0]))
                    annotations.append(Skeleton(elements, label=0)) # 0 - skeleton

                items[frame_id] = DatasetItem(id=frame_id, media=Image(data=image),
                    subset=f'subset_{video_id}', annotations=annotations)

        return items


class SleapImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("*.pkg.slp")

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, ".pkg.slp", "sleap")
