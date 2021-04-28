<template>
    <div class="wrap">
      <div class="header"></div>
      <div class="content">
        <div class="content-left">
        <img class="welcom-img" src="../assets/welcome2.png" alt="欢迎图片">
        </div>
        <div class="content-right">
          <div 
            class="card-wrap"
            v-for="(func,index) in allFunc"
            :key="index"
            >
            <a-card hoverable @click="handleClick(func.route)">
              <img
                class="card-img"
                slot="cover"
                alt="example"
                :src="func.imgurl"
              />
              <a-card-meta :title="func.title" :description="func.description"></a-card-meta>
            </a-card>
          </div>
        </div>
      </div>
      <div class="floor">
        <p>保护动物就是保护人类自己！</p>
        <p>作者：东华大学信息科学与技术学院通信专业2017级学生田翩   指导老师：廖小飞</p>
        <p>本设计现阶段仅用于毕设，不用做各种商业用途,如需参考,可联系qq：893237680</p>
      </div>
    </div>
</template>
<script>
export default {
  name: 'Home',
  data () {
    return {
      allFunc:[
        {title: "图像降噪", imgurl: require(`@/assets/reduction_title.png`),description: "大小256*256以内的图片进行降噪",route:"/reduction"},
        {title: "动物识别", imgurl: require(`@/assets/classify_title.png`),description: "不知道动物的种类，使用此功能进行识别",route:"/classify"},
      ],
    }
  },
  methods:{
    handleClick(route){
      this.$router.push({ path:route})
    }
  }

}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.wrap{
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  align-items: center;
  overflow: auto;
  background: linear-gradient(#fbfdff,#e7f3ff);
  padding: 40px 0;
}
.content{
  order:1;
  height: 75vh;
  width: 1000px;
  min-width: 1000px;
  margin: auto;
  background: #fff;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
}
.floor{
  order: 2;
  width: 100%;
  height: 7vh;
  background: rgba(129, 214, 141,0.05);
}
.content-left{
  order: 1;
  height: auto;
  width: 600px;
}
.content-right{
  order: 2;
  height: 75vh;
  width: 400px;
  background: #fff;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.welcom-img{
  height: 75vh;
  width: 600px;
}
.card-wrap{
  width: 400px;
}
.card-text{
  height: 4vh;
}
.card-img{
  overflow: hidden;
  height: 27vh;
}
p{
  margin: 1px 0;  
}
</style>