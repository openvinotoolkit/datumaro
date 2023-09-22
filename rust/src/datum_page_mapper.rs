//  Copyright (C) 2023 Intel Corporation
//
//  SPDX-License-Identifier: MIT

use std::{
    fs::File,
    io::{self, BufReader, Read, Seek},
    path::Path,
    str::FromStr,
};
use strum::EnumString;

use crate::{
    page_mapper::{JsonPageMapper, ParsedJsonSection},
    page_maps::{ImgPageMap, JsonDict},
    utils::{convert_to_py_object, invalid_data, parse_serde_json_value, read_skipping_ws},
};
use pyo3::prelude::*;
use serde_json::json;
#[derive(EnumString, Debug)]
pub enum DatumJsonSection {
    #[strum(ascii_case_insensitive)]
    DM_FORMAT_VERSION(String),
    #[strum(ascii_case_insensitive)]
    MEDIA_TYPE(i64),
    #[strum(ascii_case_insensitive)]
    INFOS(JsonDict),
    #[strum(ascii_case_insensitive)]
    CATEGORIES(JsonDict),
    #[strum(ascii_case_insensitive)]
    ITEMS(ImgPageMap<String>),
}

impl ParsedJsonSection for DatumJsonSection {
    fn parse(
        buf_key: String,
        mut reader: impl Read + Seek,
    ) -> Result<Box<DatumJsonSection>, io::Error> {
        match DatumJsonSection::from_str(buf_key.as_str()) {
            Ok(curr_key) => {
                while let Ok(c) = read_skipping_ws(&mut reader) {
                    if c == b':' {
                        break;
                    }
                }
                match curr_key {
                    DatumJsonSection::DM_FORMAT_VERSION(_) => {
                        let v = parse_serde_json_value(reader)?
                            .as_str()
                            .ok_or(invalid_data(
                                "Cannot parse datumaro format version from the json file",
                            ))?
                            .to_string();
                        Ok(Box::new(DatumJsonSection::DM_FORMAT_VERSION(v)))
                    }
                    DatumJsonSection::MEDIA_TYPE(_) => {
                        let v = parse_serde_json_value(reader)?
                            .as_i64()
                            .ok_or(invalid_data("Cannot parse media type from the json file"))?;
                        Ok(Box::new(DatumJsonSection::MEDIA_TYPE(v)))
                    }
                    DatumJsonSection::INFOS(_) => {
                        let v = parse_serde_json_value(reader)?;
                        Ok(Box::new(DatumJsonSection::INFOS(v)))
                    }
                    DatumJsonSection::CATEGORIES(_) => {
                        let v = parse_serde_json_value(reader)?;
                        Ok(Box::new(DatumJsonSection::CATEGORIES(v)))
                    }
                    DatumJsonSection::ITEMS(_) => {
                        let v = ImgPageMap::from_reader(reader)?;
                        Ok(Box::new(DatumJsonSection::ITEMS(v)))
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
pub struct DatumPageMapperImpl {
    dm_format_version: Option<String>,
    media_type: Option<i64>,
    infos: JsonDict,
    categories: JsonDict,
    items: ImgPageMap<String>,
}

impl JsonPageMapper<DatumJsonSection> for DatumPageMapperImpl {}

impl DatumPageMapperImpl {
    pub fn dm_format_version(&self) -> &Option<String> {
        return &self.dm_format_version;
    }
    pub fn media_type(&self) -> &Option<i64> {
        return &self.media_type;
    }
    pub fn infos(&self) -> &JsonDict {
        return &self.infos;
    }
    pub fn categories(&self) -> &JsonDict {
        return &self.categories;
    }
    pub fn get_img_ids(&self) -> &Vec<String> {
        self.items.ids()
    }
    pub fn get_item_dict(
        &self,
        img_id: &String,
        mut reader: impl Read + Seek,
    ) -> Result<JsonDict, io::Error> {
        self.items.get_dict(&mut reader, img_id)
    }

    pub fn new(mut reader: impl Read + Seek) -> Result<Self, io::Error> {
        let sections = Self::parse_json(&mut reader)?;

        let mut dm_format_version = None;
        let mut media_type = None;
        let mut infos = None;
        let mut categories = None;
        let mut items = None;

        for section in sections {
            match *section {
                DatumJsonSection::DM_FORMAT_VERSION(v) => {
                    dm_format_version = Some(v);
                }
                DatumJsonSection::MEDIA_TYPE(v) => {
                    media_type = Some(v);
                }
                DatumJsonSection::INFOS(v) => {
                    infos = Some(v);
                }
                DatumJsonSection::CATEGORIES(v) => {
                    categories = Some(v);
                }
                DatumJsonSection::ITEMS(v) => {
                    items = Some(v);
                }
            }
        }
        let infos = infos.unwrap_or(json!({}));
        let categories = categories.ok_or(invalid_data("Cannot find the categories section."))?;
        let items = items.ok_or(invalid_data("Cannot find the items section."))?;

        Ok(DatumPageMapperImpl {
            dm_format_version,
            media_type,
            infos,
            categories,
            items,
        })
    }
}

#[pyclass]
pub struct DatumPageMapper {
    reader: BufReader<File>,
    mapper: DatumPageMapperImpl,
}

#[pymethods]
impl DatumPageMapper {
    #[new]
    fn py_new(path: String) -> PyResult<Self> {
        let file = File::open(Path::new(&path))?;
        let mut reader = BufReader::new(file);
        let mapper = DatumPageMapperImpl::new(&mut reader)?;

        Ok(DatumPageMapper { reader, mapper })
    }

    fn dm_format_version(self_: PyRef<Self>) -> Option<String> {
        self_.mapper.dm_format_version().clone()
    }

    fn media_type(self_: PyRef<Self>) -> Option<i64> {
        self_.mapper.media_type().clone()
    }

    fn infos(self_: PyRef<Self>) -> PyResult<PyObject> {
        convert_to_py_object(self_.mapper.infos(), self_.py())
    }

    fn categories(self_: PyRef<Self>) -> PyResult<PyObject> {
        convert_to_py_object(self_.mapper.categories(), self_.py())
    }

    fn get_item_dict(&mut self, py: Python<'_>, img_id: String) -> PyResult<PyObject> {
        let item_dict = self.mapper.get_item_dict(&img_id, &mut self.reader)?;
        Ok(convert_to_py_object(&item_dict, py)?)
    }

    fn get_img_ids(&self) -> Vec<String> {
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
        const EXAMPLE: &str = r#"{"dm_format_version": "1.0", "media_type": 2, "infos": {"string": "test", "int": 0, "float": 0.0, "string_list": ["test0", "test1", "test2"], "int_list": [0, 1, 2], "float_list": [0.0, 0.1, 0.2]}, "categories": {"label": {"labels": [{"name": "cat0", "parent": "", "attributes": ["x", "y"]}, {"name": "cat1", "parent": "", "attributes": ["x", "y"]}, {"name": "cat2", "parent": "", "attributes": ["x", "y"]}, {"name": "cat3", "parent": "", "attributes": ["x", "y"]}, {"name": "cat4", "parent": "", "attributes": ["x", "y"]}], "label_groups": [], "attributes": ["a", "b", "score"]}, "mask": {"colormap": [{"label_id": 0, "r": 0, "g": 0, "b": 0}, {"label_id": 1, "r": 128, "g": 0, "b": 0}, {"label_id": 2, "r": 0, "g": 128, "b": 0}, {"label_id": 3, "r": 128, "g": 128, "b": 0}, {"label_id": 4, "r": 0, "g": 0, "b": 128}]}, "points": {"items": [{"label_id": 0, "labels": ["cat1", "cat2"], "joints": [[0, 1]]}, {"label_id": 1, "labels": ["cat1", "cat2"], "joints": [[0, 1]]}, {"label_id": 2, "labels": ["cat1", "cat2"], "joints": [[0, 1]]}, {"label_id": 3, "labels": ["cat1", "cat2"], "joints": [[0, 1]]}, {"label_id": 4, "labels": ["cat1", "cat2"], "joints": [[0, 1]]}]}}, "items": [{"id": "42", "annotations": [{"id": 900100087038, "type": "mask", "attributes": {}, "group": 900100087038, "label_id": null, "rle": {"counts": "06", "size": [2, 3]}, "z_order": 0}, {"id": 900100087038, "type": "mask", "attributes": {}, "group": 900100087038, "label_id": null, "rle": {"counts": "06", "size": [2, 3]}, "z_order": 0}], "image": {"path": "42.jpg", "size": [10, 6]}}, {"id": "43", "annotations": [], "image": {"path": "43.qq", "size": [2, 4]}}]}
        "#;

        let (tempfile, mut reader) = prepare_reader(EXAMPLE);
        let datum_page_mapper = DatumPageMapperImpl::new(&mut reader).unwrap();

        println!("{:?}", datum_page_mapper);

        let expected_ids = vec!["42".to_string(), "43".to_string()];
        assert_eq!(datum_page_mapper.get_img_ids(), &expected_ids);

        for img_id in expected_ids {
            let item = datum_page_mapper
                .get_item_dict(&img_id.to_string(), &mut reader)
                .unwrap();

            assert_eq!(item["id"].as_str(), Some(img_id.as_str()));
            println!("{:?}", item);
        }
    }
}
