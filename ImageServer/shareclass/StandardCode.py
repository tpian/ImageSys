from enum import Enum

class StandardCode(Enum):
    SUCCESS = "000000", "成功"
    PARAM_NULL = "000001", "参数为空"
    PARAM_ERROR = "000002", "参数错误"
    PARAM_NOT_DIGITAL = "000003", "参数非数字"
    PARAM_GT_0 = "000004", "参数大于0"
    PARAM_LT_0 = "000005", "参数小于0"
    PARAM_EMPTY = "000006", "参数为空"
    DATA_NOT_EXIST = "000100", "数据不存在"
    DATA_IS_EXIST = "000101", "数据已存在"
    DB_ERROR = "000102", "数据异常"
    DUPLICATE_OPERATION = "000200", "重复操作"
    SERIALIZATION_EXCEPTION = "000201", "序列化操作异常"
    INSTANTIATION_ERROR = "000202", "实例化异常"
    CLASS_CAST_ERROR = "000203", "类型转换异常"
    INVALID_TOKEN = "000401", "无效token"
    PERMISSION_DENIED = "000402", "权限不足"
    NOT_FOUND = "000404", "路径未发现"
    UNKNOWN = "000999", "网络异常"
    EXCEPTION = "000500", "操作异常"
    REMOTE_ERROR = "000300", "远程服务异常"

    def __init__(self,code,msg):
        self.code = code
        self.msg = msg