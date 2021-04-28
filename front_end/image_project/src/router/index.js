import Vue from 'vue'
import Router from 'vue-router'
import Home from '@/components/Home'
import Reduction from '@/components/Reduction'
import Classification from '@/components/Classification'


Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      redirect: '/home'
    },
    {
      path: '/home',
      name: 'Home',
      component: Home
    },
    {
      path: '/reduction',
      name: 'Reduction',
      component: Reduction
    },
    {
      path: '/classify',
      name: 'Classify',
      component: Classification
    },
    

  ]
})
