# docker

## 1.docker用来解决什么问题？
- 问题引入：
> 比如我写了一个web应用，并且本地调试没有任何问题
![web网站](assets/web应用.png)
这时候我想发给别人看看，或者部署到远程的服务器，那么就需要部署完全相同的软件，比如数据库,web服务器,必要插件和库等等，而且还不能保证一定能运行起来，而且别人还不一定用的同一个操作系统，而且就算同样的操作系统不同版本也会有差别
![sameConfig](assets/相同配置.png)

- 虚拟机的局限性
> 虚拟机需要模拟硬件,提及臃肿，内存占用高，且影响程序性能

---
这时候docker就派上了用场，它不会去模拟硬件（因此更加轻量），而是为每个应用提供完全隔离的应用环境
![dockerUsage](assets/docker作用.png)
我们可以在环境中配置不同的工具软件，并且相互不影响。这个环境叫**容器**。
具体对比如下：
![adv](assets/advantage.png)
## 2.docker三个重要概念
- **image/镜像**
> 好比虚拟机的快照snapshot，包含了要部署应用程序及所有关联的库和软件,相当于容器创建的*模板*。一个image可以创建多个互不影响的容器。

![image](assets/image.png)
- **container/容器**
> 和image的关系有点类似类和实例之间的关系，image是类，container是实例，是应用运行的环境。可以看作一个轻量级的虚拟机。
- **dockerfile**
> 一个自动化的脚本，用来创建镜像，类似于类的构造函数，这个过程好比在虚拟机中安装软件和操作系统。
## 3.docker实操
    准备工作：windows下请下载docker desktop
### Start
- 1.  下载一个用于实践的项目,在本地任找一个目录打开git bash输入下面的命令
```shell
$ git clone https://github.com/docker/getting-started.git
```
- 2. 在任意编辑器打开getting-started项目，在 app 目录（与 package.json 文件相同的位置）中，创建一个名为 Dockerfile 的文件
![](assets\df.png)
- 3.在里面写下：
```
# syntax=docker/dockerfile:1
   
FROM node:18-alpine
WORKDIR /app
COPY . .
RUN yarn install --production
CMD ["node", "src/index.js"]
EXPOSE 3000
```
然后在终端cd到app文件夹下运行：
```shell
$ docker build -t getting-started .
```
![](assets\gt.png)
可以在docker desktop的image里看到如下：
![](assets\gs.png)
- 4.  在项目终端输入，即把容器内的3000端口暴露到本地的3080端口
```shell
$ docker run -dp 3080:3000 getting-started
```
- 5.访问localhost：3080
![](assets\lc.png)
### update code
参考[updating code](https://docs.docker.com/get-started/03_updating_app/)
### share the application ...and all the rest part
参考它的其余部分，写的真的很详细~~，不需要赘述