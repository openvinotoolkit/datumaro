use crate::utils::{invalid_data, read_skipping_ws};
use std::io::{Error, Read, Seek};

pub trait ParsedJsonSection: Sized {
    fn parse(buf_key: String, reader: impl Read + Seek) -> Result<Box<Self>, Error>;
}

pub trait JsonPageMapper<T>: Sized
where
    T: ParsedJsonSection,
{
    fn parse_json(mut reader: impl Read + Seek) -> Result<Vec<Box<T>>, Error> {
        let mut brace_level = 0;
        let mut json_sections = Vec::new();

        while let Ok(c) = read_skipping_ws(&mut reader) {
            match c {
                b'{' => brace_level += 1,
                b'"' => {
                    let mut buf_key = Vec::new();
                    while let Ok(c) = read_skipping_ws(&mut reader) {
                        if c == b'"' {
                            break;
                        }
                        buf_key.push(c);
                    }
                    match String::from_utf8(buf_key.clone()) {
                        Ok(key) => {
                            let section = T::parse(key, &mut reader)?;
                            json_sections.push(section);
                        }
                        Err(e) => {
                            let cur_pos = reader.stream_position()?;
                            let msg = format!(
                                "Section key buffer, {:?} is invalid at pos: {}. {}",
                                buf_key, cur_pos, e
                            );
                            let err = invalid_data(msg.as_str());
                            return Err(err);
                        }
                    }
                }
                b',' => {
                    continue;
                }
                b'}' => {
                    brace_level -= 1;
                    if brace_level == 0 {
                        break;
                    }
                }
                _ => {
                    let cur_pos = reader.stream_position()?;
                    let msg = format!("{} is invalid character at pos: {}", c, cur_pos);
                    let err = invalid_data(msg.as_str());
                    return Err(err);
                }
            }
        }
        Ok(json_sections)
    }
}
