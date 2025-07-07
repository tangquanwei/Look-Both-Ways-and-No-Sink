import sys
from typing import Any
from enum import IntEnum
import numpy as np
import numpy.typing as npt

class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    Q4_0_4_4 = 31
    Q4_0_4_8 = 32
    Q4_0_8_8 = 33

def check_version(version):
    return version == 1 or version == 2 or version == 3

def data_get(data, offset: int, dtype: npt.DTypeLike, count: int = 1) -> npt.NDArray[Any]:
    count = int(count)
    itemsize = int(np.empty([], dtype=dtype).itemsize)
    end_offs = offset + itemsize * count
    return (data[offset:end_offs].view(dtype=dtype)[:count])

def data_read_version_size(data, offset: int, version: int):
    if version == 1:
        return data_get(data, offset, np.uint32)[0], 4
    elif version == 2 or version == 3:
        return data_get(data, offset, np.uint64)[0], 8
    else:
        raise ValueError(f'Sorry, file appears to be version {version} which we cannot handle')

def data_read_string(data, offset: int, version: int):
    str_length, str_length_len = data_read_version_size(data, offset, version)
    byte = data[offset+int(str_length_len):offset+int(str_length_len)+int(str_length)]
    value = byte.tobytes().decode('utf-8')
    len = int(str_length_len + str_length)
    return value, len

def readMetadataValue(data, type, offset, version):
    if type == GGUFValueType.UINT8:
        return data_get(data, offset, np.uint8)[0], 1
    elif type == GGUFValueType.INT8:
        return data_get(data, offset, np.int8)[0], 1
    elif type == GGUFValueType.UINT16:
        return data_get(data, offset, np.uint16)[0], 2
    elif type == GGUFValueType.INT16:
        return data_get(data, offset, np.int16)[0], 2
    elif type == GGUFValueType.UINT32:
        return data_get(data, offset, np.uint32)[0], 4
    elif type == GGUFValueType.INT32:
        return data_get(data, offset, np.int32)[0], 4
    elif type == GGUFValueType.FLOAT32:
        return data_get(data, offset, np.float32)[0], 4
    elif type == GGUFValueType.BOOL:
        return data_get(data, offset, np.uint8)[0], 1
    elif type == GGUFValueType.STRING:
        return data_read_string(data, offset, version=version)
    elif type == GGUFValueType.ARRAY:
        typeArray = data_get(data, offset, np.uint32)
        typeLength = 4
        lengthArray, lengthLength = data_read_version_size(data, offset + typeLength, version=version)
        length = typeLength + lengthLength
        arrayValues = []
        for i in range(lengthArray):
            value, len = readMetadataValue(data, typeArray, offset=offset + length, version=version)
            arrayValues.append(value)
            length += len
        return arrayValues, length
    elif type == GGUFValueType.UINT64:
        return data_get(data, offset, np.uint64)[0], 8
    elif type == GGUFValueType.INT64:
        return data_get(data, offset, np.int64)[0], 8
    elif type == GGUFValueType.FLOAT64:
        return data_get(data, offset, np.float64)[0], 8
    else:
        raise ValueError(f'Sorry, un-supported GGUFValueType {type}!')

def parse_gguf(model_path):
    data = np.memmap(model_path, mode='r')
    offs = 0
    magic = data_get(data, offs, np.uint32).tobytes()
    print("magic: ", magic.decode('utf-8'))
    if (magic != b'GGUF'):
        print("is not gguf file")
        sys.exit(1)

    offs += 4
    version = data_get(data, offs, np.uint32)
    if not check_version(version):
        raise ValueError(f'Sorry, file appears to be version {version} which we cannot handle')
    print("version:", version)

    offs += 4
    tensor_count, tensor_count_len = data_read_version_size(data, offs, version)
    offs += tensor_count_len
    kv_count, kv_count_len = data_read_version_size(data, offs, version)
    offs += kv_count_len
    print("tensor_count: ", tensor_count)
    print("kv_count: ", kv_count)

    metadata = {}
    for i in range(kv_count):
        key, k_len = data_read_string(data, offs, version)
        offs += k_len
        type = data_get(data, offs, np.uint32)[0]
        offs += 4
        value, len = readMetadataValue(data, type, offs, version)
        if len > 100:
            print(f"[{i}]", key, ":", value[:100])
        else:
            print(f"[{i}]", key, ":", value)
        offs += len
        metadata[key] = value

    for i in range(tensor_count):
        key, k_len = data_read_string(data, offs, version)
        offs += k_len
        nDims = data_get(data, offs, np.uint32)[0]
        offs += 4
        dims = []
        for _ in range(nDims):
            dim, dim_len = data_read_version_size(data, offs, version)
            offs += dim_len
            dims.append(dim)
        types = data_get(data, offs, np.uint32)[0]
        precision = GGMLQuantizationType(types).name
        offs += 4
        tensorOffset = data_get(data, offs, np.uint64)[0]
        offs += 8
        print(f"tensor [{i}]", "k = ", key, ", precision = ", precision, ", shape = ", dims, ", tensorOffset = ", tensorOffset)

if __name__ == '__main__':
    model_path = sys.argv[1]
    parse_gguf(model_path)