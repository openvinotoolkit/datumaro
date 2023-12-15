//  Copyright (C) 2023 Intel Corporation
//
//  SPDX-License-Identifier: MIT
mod test_helpers;

use std::io::{self, Seek, Read};
use datumaro_rust_api::json_section_page_mapper::JsonSectionPageMapperImpl;
use test_helpers::prepare_reader;

#[test]
fn test_instance() {
    const EXAMPLE: &str = r#"{"dm_format_version": "1.0", "media_type": 2, "infos": {"string": "test", "int": 0, "float": 0.0, "string_list": ["test0", "test1", "test2"], "int_list": [0, 1, 2], "float_list": [0.0, 0.1, 0.2]}, "categories": {"label": {"labels": [{"name": "cat0", "parent": "", "attributes": ["x", "y"]}, {"name": "cat1", "parent": "", "attributes": ["x", "y"]}, {"name": "cat2", "parent": "", "attributes": ["x", "y"]}, {"name": "cat3", "parent": "", "attributes": ["x", "y"]}, {"name": "cat4", "parent": "", "attributes": ["x", "y"]}], "label_groups": [], "attributes": ["a", "b", "score"]}, "mask": {"colormap": [{"label_id": 0, "r": 0, "g": 0, "b": 0}, {"label_id": 1, "r": 128, "g": 0, "b": 0}, {"label_id": 2, "r": 0, "g": 128, "b": 0}, {"label_id": 3, "r": 128, "g": 128, "b": 0}, {"label_id": 4, "r": 0, "g": 0, "b": 128}]}, "points": {"items": [{"label_id": 0, "labels": ["cat1", "cat2"], "joints": [[0, 1]]}, {"label_id": 1, "labels": ["cat1", "cat2"], "joints": [[0, 1]]}, {"label_id": 2, "labels": ["cat1", "cat2"], "joints": [[0, 1]]}, {"label_id": 3, "labels": ["cat1", "cat2"], "joints": [[0, 1]]}, {"label_id": 4, "labels": ["cat1", "cat2"], "joints": [[0, 1]]}]}}, "items": [{"id": "42", "annotations": [{"id": 900100087038, "type": "mask", "attributes": {}, "group": 900100087038, "label_id": null, "rle": {"counts": "06", "size": [2, 3]}, "z_order": 0}, {"id": 900100087038, "type": "mask", "attributes": {}, "group": 900100087038, "label_id": null, "rle": {"counts": "06", "size": [2, 3]}, "z_order": 0}], "image": {"path": "42.jpg", "size": [10, 6]}}, {"id": "43", "annotations": [], "image": {"path": "43.qq", "size": [2, 4]}}]}
    "#;

    let (tempfile, mut reader) = prepare_reader(EXAMPLE);
    let json_section_page_mapper = JsonSectionPageMapperImpl::new(&mut reader).unwrap();

    println!("{:?}", json_section_page_mapper);

    for section in json_section_page_mapper.sections {
        let offset = section.offset;
        let size = section.size;
        reader.seek(io::SeekFrom::Start(offset as u64));
        let mut buf = vec![0; size];
        reader.read(buf.as_mut_slice());

        let content: serde_json::Value = serde_json::from_str(
            std::str::from_utf8(buf.as_slice()).expect("Cannot change to utf8"),
        )
        .unwrap();
        println!("Section: {}, Content: {:?}", section.key, content);
    }
}
