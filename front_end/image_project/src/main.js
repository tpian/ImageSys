// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import antd from './components/antd';
import VueAxios from 'vue-axios'
import axios from 'axios'
// import {httpGet, httpPost} from './axios';

Vue.config.productionTip = false
Vue.use(antd);
Vue.use(VueAxios, axios)
// Vue.prototype.$httpGet = httpGet;
// Vue.prototype.$httpPost = httpPost;
/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
