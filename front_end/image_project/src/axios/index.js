import qstring from "qs";
import axios from "axios";
import { Modal } from "ant-design-vue";
let axiosobj = axios.create({
  withCredentials: true,
  baseURL: "/ctu_api"
});
// http request 拦截器
axiosobj.interceptors.request.use(
  config => {
    if (config.method === "post") {
      if (config.headers.type == "file") {
        config.headers["Content-Type"] = "multipart/form-data";
      } else if (config.headers.type == "form") {
        config.data = qstring.stringify(config.data);
        config.headers["Content-Type"] = "application/x-www-form-urlencoded";
      } else {
        config.data = JSON.stringify(config.data);
        config.headers["Content-Type"] = "application/json;charset=utf-8";
      }
    }
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// http response 拦截器
axiosobj.interceptors.response.use(
  response => {
    let { status, data = {} } = response || {};
    if (data.code == "000401") {
      Modal.destroyAll(); // 防止调用多次接口，弹出多个确认框
      Modal.warning({
        title: "系统提示",
        content: "登录失效，请重新登录",
        okText: "确定",
        onOk: () => {
          window.location.href = "/main/login";
        }
      });
      return false;
    }
    if (status === 200) {
      return Promise.resolve(data);
    }
    return Promise.reject(data);
  },
  error => {
    return Promise.reject(error);
  }
);
/**
 * get请求
 * @param {String} url 请求地址
 * @param {Object} params 请求参数
 */
export function httpGet(url, params = {}, config = {}) {
  return new Promise((resolve, reject) => {
    axiosobj
      .get(url, { ...config, params })
      .then(res => {
        resolve(res);
      })
      .catch(err => {
        reject(err);
      });
  });
}

/**
 * post请求
 * @param {String} url 请求地址
 * @param {Object} params 请求参数
 */
export function httpPost(url, params = {}, config = {}) {
  let { headers = {}, ...surconfig } = config || {};
  return new Promise((resolve, reject) => {
    axiosobj
      .post(url, params, { ...surconfig, headers: { ...headers } })
      .then(res => {
        resolve(res);
      })
      .catch(err => {
        reject(err);
      });
  });
}
