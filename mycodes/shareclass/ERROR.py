from Response import Response
from StandardCode import StandardCode

class ERROR():
    PARAM_NULL = Response.error1(StandardCode.PARAM_NULL.value[0], StandardCode.PARAM_NULL.msg)
    PARAM_ERROR = Response.error1(StandardCode.PARAM_ERROR.value[0], StandardCode.PARAM_ERROR.value[1])
    PARAM_NOT_DIGITAL = Response.error1(StandardCode.PARAM_NOT_DIGITAL.value[0],
                                        StandardCode.PARAM_NOT_DIGITAL.value[1])
    PARAM_GT_0 = Response.error1(StandardCode.PARAM_GT_0.value[0], StandardCode.PARAM_GT_0.value[1])
    PARAM_LT_0 = Response.error1(StandardCode.PARAM_LT_0.value[0], StandardCode.PARAM_LT_0.value[1])
    PARAM_EMPTY = Response.error1(StandardCode.PARAM_EMPTY.value[0], StandardCode.PARAM_EMPTY.value[1])
    DATA_NOT_EXIST = Response.error1(StandardCode.DATA_NOT_EXIST.value[0], StandardCode.DATA_NOT_EXIST.value[1])
    DATA_IS_EXIST = Response.error1(StandardCode.DATA_IS_EXIST.value[0], StandardCode.DATA_IS_EXIST.value[1])
    DB_ERROR = Response.error1(StandardCode.DB_ERROR.value[0], StandardCode.DB_ERROR.value[1])
    DUPLICATE_OPERATION = Response.error1(StandardCode.DUPLICATE_OPERATION.value[0],
                                          StandardCode.DUPLICATE_OPERATION.value[1])
    SERIALIZATION_EXCEPTION = Response.error1(StandardCode.SERIALIZATION_EXCEPTION.value[0],
                                              StandardCode.SERIALIZATION_EXCEPTION.value[1])
    INSTANTIATION_ERROR = Response.error1(StandardCode.INSTANTIATION_ERROR.value[0], StandardCode.DB_ERROR.value[1])
    CLASS_CAST_ERROR = Response.error1(StandardCode.CLASS_CAST_ERROR.value[0],
                                       StandardCode.INSTANTIATION_ERROR.value[1])
    INVALID_TOKEN = Response.error1(StandardCode.INVALID_TOKEN.value[0], StandardCode.INVALID_TOKEN.value[1])
    PERMISSION_DENIED = Response.error1(StandardCode.PERMISSION_DENIED.value[0],
                                        StandardCode.PERMISSION_DENIED.value[1])
    NOT_FOUND = Response.error1(StandardCode.NOT_FOUND.value[0], StandardCode.NOT_FOUND.value[1])
    UNKNOWN = Response.error1(StandardCode.UNKNOWN.value[0], StandardCode.UNKNOWN.value[1])
    REMOTE_ERROR = Response.error1(StandardCode.REMOTE_ERROR.value[0], StandardCode.REMOTE_ERROR.value[1])
    EXCEPTION = Response.error1(StandardCode.EXCEPTION.value[0], StandardCode.EXCEPTION.value[1])