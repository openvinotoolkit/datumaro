# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ava_label.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ava_label.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0f\x61va_label.proto\"\xa0\x01\n\x05Label\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x10\n\x08label_id\x18\x02 \x01(\x05\x12$\n\nlabel_type\x18\x03 \x01(\x0e\x32\x10.Label.LabelType\"Q\n\tLabelType\x12\x13\n\x0fPERSON_MOVEMENT\x10\x00\x12\x17\n\x13OBJECT_MANIPULATION\x10\x01\x12\x16\n\x12PERSON_INTERACTION\x10\x02\"\"\n\tLabelList\x12\x15\n\x05label\x18\x01 \x03(\x0b\x32\x06.Labelb\x06proto3')
)



_LABEL_LABELTYPE = _descriptor.EnumDescriptor(
  name='LabelType',
  full_name='Label.LabelType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PERSON_MOVEMENT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBJECT_MANIPULATION', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PERSON_INTERACTION', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=99,
  serialized_end=180,
)
_sym_db.RegisterEnumDescriptor(_LABEL_LABELTYPE)


_LABEL = _descriptor.Descriptor(
  name='Label',
  full_name='Label',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='Label.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label_id', full_name='Label.label_id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label_type', full_name='Label.label_type', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _LABEL_LABELTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=20,
  serialized_end=180,
)


_LABELLIST = _descriptor.Descriptor(
  name='LabelList',
  full_name='LabelList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='label', full_name='LabelList.label', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=182,
  serialized_end=216,
)

_LABEL.fields_by_name['label_type'].enum_type = _LABEL_LABELTYPE
_LABEL_LABELTYPE.containing_type = _LABEL
_LABELLIST.fields_by_name['label'].message_type = _LABEL
DESCRIPTOR.message_types_by_name['Label'] = _LABEL
DESCRIPTOR.message_types_by_name['LabelList'] = _LABELLIST
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Label = _reflection.GeneratedProtocolMessageType('Label', (_message.Message,), dict(
  DESCRIPTOR = _LABEL,
  __module__ = 'ava_label_pb2'
  # @@protoc_insertion_point(class_scope:Label)
  ))
_sym_db.RegisterMessage(Label)

LabelList = _reflection.GeneratedProtocolMessageType('LabelList', (_message.Message,), dict(
  DESCRIPTOR = _LABELLIST,
  __module__ = 'ava_label_pb2'
  # @@protoc_insertion_point(class_scope:LabelList)
  ))
_sym_db.RegisterMessage(LabelList)


# @@protoc_insertion_point(module_scope)
