//  Copyright (C) 2023 Intel Corporation
//
//  SPDX-License-Identifier: MIT

use std::io::{self};

pub fn read_skipping_ws(mut reader: impl io::Read) -> io::Result<u8> {
    loop {
        let mut byte = 0u8;
        reader.read_exact(std::slice::from_mut(&mut byte))?;
        if !byte.is_ascii_whitespace() {
            return Ok(byte);
        }
    }
}

pub fn invalid_data(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

pub fn stream_error(error: &str, offset: u64) -> io::Error {
    let msg = format!("[Parse error, offset={}] {}", offset, error);
    invalid_data(msg.as_str())
}

pub fn parse_serde_json_value_from_page<R>(
    reader: &mut R,
    offset: u64,
    size: u64,
) -> Result<serde_json::Value, io::Error>
where
    R: io::Read + io::Seek,
{
    reader.seek(io::SeekFrom::Start(offset))?;

    let mut buf = vec![0u8; size as usize];
    let _ = reader.read(buf.as_mut_slice())?;

    let img_dict_str = String::from_utf8(buf).ok().ok_or(invalid_data(
        format!("Cannot read offset: {} and size: {}", offset, size).as_str(),
    ))?;

    serde_json::from_str(img_dict_str.as_str())
        .ok()
        .ok_or(invalid_data(
            format!("Cannot parse to dict offset: {} and size: {}", offset, size).as_str(),
        ))
}

pub fn parse_serde_json_value(
    reader: impl io::Read + io::Seek,
) -> Result<serde_json::Value, io::Error> {
    let de = serde_json::Deserializer::from_reader(reader);
    let mut stream = de.into_iter::<serde_json::Value>();
    match stream.next().unwrap() {
        Ok(x) => Ok(x),
        Err(e) => {
            let cur_pos = stream.byte_offset();
            let msg = format!("Parse error: {} at pos: {}", e, cur_pos);
            Err(invalid_data(msg.as_str()))
        }
    }
}
