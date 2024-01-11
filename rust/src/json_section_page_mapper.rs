//  Copyright (C) 2023 Intel Corporation
//
//  SPDX-License-Identifier: MIT

use crate::{
    page_mapper::{JsonPageMapper, ParsedJsonSection},
    utils::read_skipping_ws,
};
use pyo3::{prelude::*, types::PyDict};
use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader, Read, Seek},
    path::Path,
};

#[derive(Debug)]
struct JsonSection {
    key: String,
    offset: usize,
    size: usize,
}

fn handle_arr_or_dict(
    mut stack: Vec<u8>,
    mut reader: impl Read + Seek,
    mut last_token: u8,
) -> Result<(), io::Error> {
    while stack.len() != 0 {
        match read_skipping_ws(&mut reader) {
            Ok(c) => match c {
                b'{' | b'[' => {
                    stack.push(c);
                    last_token = c;
                }
                b'}' => {
                    if last_token != b'{' {
                        let cur_pos = reader.stream_position()?;
                        let msg = format!("Last token in the stack is '{}', but the given token at offset={} is '}}'", last_token as char, cur_pos);
                        return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                    }
                    stack.pop();
                    if stack.len() != 0 {
                        last_token = *stack
                            .last()
                            .ok_or(io::Error::new(io::ErrorKind::InvalidData, "stack is empty"))?;
                    }
                }
                b']' => {
                    if last_token != b'[' {
                        let cur_pos = reader.stream_position()?;
                        let msg = format!("Last token in the stack is '{}', but the given token at offset={} is ']'", last_token as char, cur_pos);
                        return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                    }
                    stack.pop();
                    if stack.len() != 0 {
                        last_token = *stack
                            .last()
                            .ok_or(io::Error::new(io::ErrorKind::InvalidData, "stack is empty"))?;
                    }
                }
                b'"' => {
                    while let Ok(c) = read_skipping_ws(&mut reader) {
                        if c == b'"' {
                            break;
                        }
                    }
                }
                _ => {}
            },
            Err(err) => {
                return Err(err);
            }
        }
    }
    Ok(())
}

fn handle_string(mut reader: impl Read + Seek) -> Result<(), io::Error> {
    while let Ok(c) = read_skipping_ws(&mut reader) {
        if c == b'"' {
            break;
        }
    }
    Ok(())
}

fn get_offset(mut reader: impl Read + Seek, stack: &mut Vec<u8>) -> Result<usize, io::Error> {
    let mut offset = usize::MAX;
    while let Ok(c) = read_skipping_ws(&mut reader) {
        stack.push(c);
        match c {
            b'{' | b'[' | b'"' => {
                return Ok(reader.stream_position()? as usize - 1);
            }
            b',' => {
                return Ok(offset - 1);
            }
            _ => {
                let pos = reader.stream_position()? as usize;
                offset = std::cmp::min(pos, offset);
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "Cannot get offset",
    ))
}

impl ParsedJsonSection for JsonSection {
    fn parse(buf_key: String, mut reader: impl Read + Seek) -> Result<Box<JsonSection>, io::Error> {
        // Move reader's cursor right after ':'
        while let Ok(c) = read_skipping_ws(&mut reader) {
            if c == b':' {
                break;
            }
        }

        let mut stack = vec![];

        let start_offset = get_offset(&mut reader, &mut stack)?;

        let last_token = *stack
            .last()
            .ok_or(io::Error::new(io::ErrorKind::InvalidData, "stack is empty"))?;

        let end_offset = match last_token {
            b'[' | b'{' => {
                let _ = handle_arr_or_dict(stack, &mut reader, last_token)?;
                Ok(reader.stream_position()? as usize)
            }
            b'"' => {
                let _ = handle_string(&mut reader)?;
                Ok(reader.stream_position()? as usize)
            }
            b',' => Ok(reader.stream_position()? as usize - 1),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "s")),
        }?;

        let size = end_offset - start_offset;

        Ok(Box::new(JsonSection {
            key: buf_key,
            offset: start_offset,
            size: size,
        }))
    }
}

#[derive(Debug)]
pub struct JsonSectionPageMapperImpl {
    sections: Vec<Box<JsonSection>>,
}

impl JsonPageMapper<JsonSection> for JsonSectionPageMapperImpl {}

impl JsonSectionPageMapperImpl {
    pub fn new(mut reader: impl Read + Seek) -> Result<Self, io::Error> {
        let sections = Self::parse_json(&mut reader)?;

        Ok(JsonSectionPageMapperImpl { sections: sections })
    }
}

#[pyclass]
pub struct JsonSectionPageMapper {
    reader: BufReader<File>,
    mapper: JsonSectionPageMapperImpl,
}

#[pymethods]
impl JsonSectionPageMapper {
    #[new]
    fn py_new(path: String) -> PyResult<Self> {
        let file = File::open(Path::new(&path))?;
        let mut reader = BufReader::new(file);
        let mapper = JsonSectionPageMapperImpl::new(&mut reader)?;

        Ok(JsonSectionPageMapper { reader, mapper })
    }

    fn sections(self_: PyRef<Self>) -> PyResult<PyObject> {
        let dict: HashMap<&str, HashMap<&str, usize>> = self_
            .mapper
            .sections
            .iter()
            .map(|section| {
                let nested_dict: HashMap<&str, usize> =
                    HashMap::from_iter([("offset", section.offset), ("size", section.size)]);
                (section.key.as_str(), nested_dict)
            })
            .collect();

        Ok(dict.into_py(self_.py()))
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.mapper.sections.len())
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
}
