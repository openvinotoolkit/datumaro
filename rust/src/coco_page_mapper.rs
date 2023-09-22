//  Copyright (C) 2023 Intel Corporation
//
//  SPDX-License-Identifier: MIT

use std::{
    io::{self, Read, Seek},
    str::FromStr,
};
use strum::EnumString;

use crate::{
    page_mapper::{JsonPageMapper, ParsedJsonSection},
    page_maps::{AnnPageMap, ImgPageMap, JsonDict},
    utils::{convert_to_py_object, invalid_data, parse_serde_json_value, read_skipping_ws},
};
use pyo3::{prelude::*, types::PyList};
use std::{fs::File, io::BufReader, path::Path};

#[derive(EnumString, Debug)]
enum CocoJsonSection {
    #[strum(ascii_case_insensitive)]
    LICENSES(JsonDict),
    #[strum(ascii_case_insensitive)]
    INFO(JsonDict),
    #[strum(ascii_case_insensitive)]
    CATEGORIES(JsonDict),
    #[strum(ascii_case_insensitive)]
    IMAGES(ImgPageMap<i64>),
    #[strum(ascii_case_insensitive)]
    ANNOTATIONS(AnnPageMap),
}

impl ParsedJsonSection for CocoJsonSection {
    fn parse(
        buf_key: String,
        mut reader: impl Read + Seek,
    ) -> Result<Box<CocoJsonSection>, io::Error> {
        match CocoJsonSection::from_str(buf_key.as_str()) {
            Ok(curr_key) => {
                while let Ok(c) = read_skipping_ws(&mut reader) {
                    if c == b':' {
                        break;
                    }
                }
                match curr_key {
                    CocoJsonSection::LICENSES(_) => {
                        let v = parse_serde_json_value(reader)?;
                        Ok(Box::new(CocoJsonSection::LICENSES(v)))
                    }
                    CocoJsonSection::INFO(_) => {
                        let v = parse_serde_json_value(reader)?;
                        Ok(Box::new(CocoJsonSection::INFO(v)))
                    }
                    CocoJsonSection::CATEGORIES(_) => {
                        let v = parse_serde_json_value(reader)?;
                        Ok(Box::new(CocoJsonSection::CATEGORIES(v)))
                    }
                    CocoJsonSection::IMAGES(_) => {
                        let v = ImgPageMap::from_reader(reader)?;
                        Ok(Box::new(CocoJsonSection::IMAGES(v)))
                    }
                    CocoJsonSection::ANNOTATIONS(_) => {
                        let v = AnnPageMap::from_reader(reader)?;
                        Ok(Box::new(CocoJsonSection::ANNOTATIONS(v)))
                    }
                }
            }
            Err(e) => {
                let cur_pos = reader.stream_position()?;
                let msg = format!("Unknown key: {} at pos: {}", e, cur_pos);
                Err(invalid_data(msg.as_str()))
            }
        }
    }
}

#[derive(Debug)]
struct CocoPageMapperImpl {
    licenses: JsonDict,
    info: JsonDict,
    categories: JsonDict,
    images: ImgPageMap<i64>,
    annotations: AnnPageMap,
}

impl JsonPageMapper<CocoJsonSection> for CocoPageMapperImpl {}

impl CocoPageMapperImpl {
    pub fn licenses(&self) -> &JsonDict {
        return &self.licenses;
    }
    pub fn info(&self) -> &JsonDict {
        return &self.info;
    }
    pub fn categories(&self) -> &JsonDict {
        return &self.categories;
    }
    pub fn get_img_ids(&self) -> &Vec<i64> {
        self.images.ids()
    }
    pub fn get_item_dict(
        &self,
        img_id: &i64,
        mut reader: impl Read + Seek,
    ) -> Result<JsonDict, io::Error> {
        self.images.get_dict(&mut reader, img_id)
    }
    pub fn get_anns_dict(
        &self,
        img_id: i64,
        mut reader: impl Read + Seek,
    ) -> Result<Vec<JsonDict>, io::Error> {
        self.annotations.get_anns(&mut reader, img_id)
    }
    pub fn new(mut reader: impl Read + Seek) -> Result<Self, io::Error> {
        let sections = Self::parse_json(&mut reader)?;

        let mut licenses = None;
        let mut info = None;
        let mut categories = None;
        let mut images = None;
        let mut annotations = None;

        for section in sections {
            match *section {
                CocoJsonSection::LICENSES(v) => {
                    licenses = Some(v);
                }
                CocoJsonSection::INFO(v) => {
                    info = Some(v);
                }
                CocoJsonSection::CATEGORIES(v) => {
                    categories = Some(v);
                }
                CocoJsonSection::IMAGES(v) => {
                    images = Some(v);
                }
                CocoJsonSection::ANNOTATIONS(v) => {
                    annotations = Some(v);
                }
            }
        }

        let licenses = licenses.ok_or(invalid_data("Cannot find the licenses section."))?;
        let info = info.ok_or(invalid_data("Cannot find the info section."))?;
        let categories = categories.ok_or(invalid_data("Cannot find the categories section."))?;
        let images = images.ok_or(invalid_data("Cannot find the images section."))?;
        let annotations =
            annotations.ok_or(invalid_data("Cannot find the annotations section."))?;

        Ok(CocoPageMapperImpl {
            licenses,
            info,
            categories,
            images,
            annotations,
        })
    }
}

#[pyclass]
pub struct CocoPageMapper {
    reader: BufReader<File>,
    mapper: CocoPageMapperImpl,
}

#[pymethods]
impl CocoPageMapper {
    #[new]
    fn py_new(path: String) -> PyResult<Self> {
        let file = File::open(Path::new(&path))?;
        let mut reader = BufReader::new(file);
        let mapper = CocoPageMapperImpl::new(&mut reader)?;

        Ok(CocoPageMapper { reader, mapper })
    }

    fn licenses(self_: PyRef<Self>) -> PyResult<PyObject> {
        convert_to_py_object(self_.mapper.licenses(), self_.py())
    }

    fn info(self_: PyRef<Self>) -> PyResult<PyObject> {
        convert_to_py_object(self_.mapper.info(), self_.py())
    }

    fn categories(self_: PyRef<Self>) -> PyResult<PyObject> {
        convert_to_py_object(self_.mapper.categories(), self_.py())
    }

    fn get_item_dict(&mut self, py: Python<'_>, img_id: i64) -> PyResult<PyObject> {
        let item_dict = self.mapper.get_item_dict(&img_id, &mut self.reader)?;
        Ok(convert_to_py_object(&item_dict, py)?)
    }

    fn get_anns_dict(&mut self, py: Python<'_>, img_id: i64) -> PyResult<PyObject> {
        let anns_list = PyList::new(
            py,
            self.mapper
                .get_anns_dict(img_id, &mut self.reader)?
                .iter()
                .map(|child| convert_to_py_object(child, py).unwrap()),
        );
        Ok(anns_list.into())
    }

    fn get_img_ids(&self) -> Vec<i64> {
        self.mapper.get_img_ids().to_owned()
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.mapper.get_img_ids().len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::prepare_reader;

    #[test]
    fn test_instance() {
        const EXAMPLE: &str = r#"
        {
            "licenses":[{"name":"test_instance()","id":0,"url":""}],
            "info":{"contributor":"","date_created":"","description":"","url":"","version":"","year":""},
            "categories":[
                {"id":1,"name":"a","supercategory":""},
                {"id":2,"name":"b","supercategory":""},
                {"id":4,"name":"c","supercategory":""}
            ],
            "images":[
                {"id":5,"width":10,"height":5,"file_name":"a.jpg","license":0,"flickr_url":"","coco_url":"","date_captured":0},
                {"id":6,"width":10,"height":5,"file_name":"b.jpg","license":0,"flickr_url":"","coco_url":"","date_captured":0}
            ],
            "annotations":[
                {"id":1,"image_id":5,"category_id":2,"segmentation":[],"area":3.0,"bbox":[2.0,2.0,3.0,1.0],"iscrowd":0},
                {"id":2,"image_id":5,"category_id":2,"segmentation":[],"area":3.0,"bbox":[2.0,2.0,3.0,1.0],"iscrowd":0},
                {"id":3,"image_id":5,"category_id":2,"segmentation":[],"area":3.0,"bbox":[2.0,2.0,3.0,1.0],"iscrowd":0},
                {"id":4,"image_id":6,"category_id":2,"segmentation":[],"area":3.0,"bbox":[2.0,2.0,3.0,1.0],"iscrowd":0},
                {"id":5,"image_id":6,"category_id":2,"segmentation":[],"area":3.0,"bbox":[2.0,2.0,3.0,1.0],"iscrowd":0}
            ]
        }"#;

        let (tempfile, mut reader) = prepare_reader(EXAMPLE);
        let coco_page_mapper = CocoPageMapperImpl::new(&mut reader).unwrap();

        println!("{:?}", coco_page_mapper);

        for img_id in [5, 6] {
            let item = coco_page_mapper
                .get_item_dict(&img_id, &mut reader)
                .unwrap();

            assert_eq!(item["id"].as_i64(), Some(img_id));

            let anns = coco_page_mapper.get_anns_dict(img_id, &mut reader).unwrap();
            assert!(anns.len() > 0);

            for ann in anns {
                assert_eq!(ann["image_id"].as_i64(), Some(img_id));
            }
        }
    }

    #[test]
    fn test_image_info_default() {
        const EXAMPLE: &str = r#"
        {"licenses": [{"name": "", "id": 0, "url": ""}], "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""}, "categories": [], "images": [{"id": 1, "width": 2, "height": 4, "file_name": "1.jpg", "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0}], "annotations": []}
        "#;

        let (tempfile, mut reader) = prepare_reader(EXAMPLE);
        let coco_page_mapper = CocoPageMapperImpl::new(&mut reader).unwrap();

        println!("{:?}", coco_page_mapper);
    }

    #[test]
    fn test_panoptic_has_no_ann_id() {
        const EXAMPLE: &str = r#"
        {"licenses":[{"name":"","id":0,"url":""}],"info":{"contributor":"","date_created":"","description":"","url":"","version":"","year":""},"categories":[{"id":1,"name":"0","supercategory":"","isthing":0},{"id":2,"name":"1","supercategory":"","isthing":0},{"id":3,"name":"2","supercategory":"","isthing":0},{"id":4,"name":"3","supercategory":"","isthing":0},{"id":5,"name":"4","supercategory":"","isthing":0},{"id":6,"name":"5","supercategory":"","isthing":0},{"id":7,"name":"6","supercategory":"","isthing":0},{"id":8,"name":"7","supercategory":"","isthing":0},{"id":9,"name":"8","supercategory":"","isthing":0},{"id":10,"name":"9","supercategory":"","isthing":0}],"images":[{"id":1,"width":4,"height":4,"file_name":"1.jpg","license":0,"flickr_url":"","coco_url":"","date_captured":0}],"annotations":[{"image_id":1,"file_name":"1.png","segments_info":[{"id":3,"category_id":5,"area":5.0,"bbox":[1.0,0.0,2.0,2.0],"iscrowd":0}]}]}
        "#;

        let (tempfile, mut reader) = prepare_reader(EXAMPLE);
        let coco_page_mapper = CocoPageMapperImpl::new(&mut reader).unwrap();

        println!("{:?}", coco_page_mapper);
    }
}
