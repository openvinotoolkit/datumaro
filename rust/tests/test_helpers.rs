//  Copyright (C) 2023 Intel Corporation
//
//  SPDX-License-Identifier: MIT

use std::{
    fs::File,
    io::{BufReader, Write},
};
use tempfile::NamedTempFile;

pub fn prepare_reader(example: &str) -> (NamedTempFile, BufReader<File>) {
    let mut tempfile = NamedTempFile::new().expect("cannot open file");
    let _ = tempfile.write_all(example.as_bytes());
    let f = File::open(tempfile.path()).expect("cannot open file");
    let mut reader = BufReader::new(f);

    (tempfile, reader)
}
