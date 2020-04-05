# Docker for Linux: Setup

1. Install Docker

   ~~~shell
   $ curl -sSL https://get.docker.com/ | sh
   ~~~

2. Add your user to docker group

   Let users run docker commands without typing sudo every time. 

   ~~~shell
   $ sudo usermod -aG docker sanghapark
   ~~~

   Log out and log back in to see the change.

3. Two other things: Docker Machine & Docker Compose

   - Docker Machine

     ~~~shell
     base=https://github.com/docker/machine/releases/download/v0.16.0 &&
       curl -L $base/docker-machine-$(uname -s)-$(uname -m) >/tmp/docker-machine &&
       sudo install /tmp/docker-machine /usr/local/bin/docker-machine
     ~~~

     아래와 같은 에러 발생시

     ~~~shell
     curl: (35) Peer reports incompatible or unsupported protocol version.
     ~~~

     curl을 업데이트 하자. ([참고](https://github.com/ChuanyuWang/test/wiki/How-to-Fix-Git-Fetch-error-on-CentOS))

     ~~~shell
     $ yum updae curl
     ~~~

   - Docker Compose

     ~~~shell
     $ curl -L https://github.com/docker/compose/releases/download/1.24.0-rc1/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
     $ chmod +x /usr/local/bin/docker-compose
     ~~~

     permision 에러가 발생 할 것이다. root에서 실행 해야 한다.

     ~~~shell
     $ sudo -i
     ~~~

     위에 코맨드들을 다시 실행하자.

4. Git

   ~~~shell
   sudo yum install git
   ~~~

   

# Create and Use Containers

- An image is the application we want to run

- A container is an instance of that image running as a process

- You can have many containers running off the same image

  

### Docker Info

~~~shell
# verified cli can talk to engine
$ docker version

# most config values of engine
$ docker info
~~~



### Starting a Nginx web server

We  will run the Nginx web server 

- Download image 'nginx' from Docker Hub
- Start a new container from that image
- Open up 80 on the host IP and route that trarric to the container IP, port 80

~~~shell
# run it in the background with name
$ docker container run --publish 80:80 --detach --name webhost nginx

# show logs for a specific container
$ docker container logs <CONTAINER NAME or ID>

# Display the running processes of a container
$ docker container top <CONTAINER NAME or ID>

# list running containers
$ docker container ls

# stop running container
$ docker container stop <CONTAINER ID>

# Remove exited container (add -f option to remove running container)
$ docker container rm <CONTAINER ID> ... <CONTAINER ID>
~~~



### What happens in 'docker container run'

- Look for that image locally in image cache
- If not existing locally, download from remote image repository (defaults to Docker Hub)
- Downloads the latest version (nginx:latest by default)
- Create new container based on the image and prepare to start
- Give it a virtual IP on a private network inside docker engine
- Open up port 80 on host and forward to port 80 in container
- Start conrainer by using the CMD in the image Dockerfile



### Container vs. Virtual Machine: It's just a process



### Assignment: Manage Multiple Containers

~~~shell
# nginx
$ docker container run -d -p 80:80 --detach --name proxy nginx

# mysql
$ docker container run -d -p 3306:3306 --name db -e MYSQL_RANDOM_ROOT_PASSWORD=yes mysql

# httpd
$ docker container run -d -p 8080:80 --name webserver httpd

# Find the random password of mysql
$ docker container logs db

# test
$ curl localhost:80
$ curl lopcalhost:8080

# Clean up
$ docker container stop proxy db webserver
$ docker container rm proxy db webserver

# Check nothing is running
$ docker container ls
~~~



### What's going on in containers 

~~~shell
# process list in one container
$ docker container top <CONTAINER NAME>

# details of container config
$ docker container inspect <CONTAINER NAME>

# performance stats for all containers
$ docker container stats
~~~



### Getting a Shell inside containers (No SSH needed)

~~~shell
# start new container interactively
# -it 옵션은 다음 두개의 -i -t 옵션을 합친 것
$ docker container run -it --name proxy nginx bash
~~~

- bash를 추가 하지 않았으면 기본적으로 "nginx -g 'daemon …" 커맨드를 실행한다. bash로 들어 가서 exit을 통해 밖으로 나오면 컨테이너는 stop 한다.



~~~shell
# run additional command in existing container
$ docker container exec -it mysql bash 
~~~

- exit해도 컨테이너는 계속 run한다.



### Docker Networks: Private and Public Communication for Containers

- Each container connected to a private virtual network "bridge"
- Each virtual network routes through NAT firewall on host IP
- All containers on a virtual network can talk to each other without -p
- Best practice is to create a new virtual network for each app
  - mysql + php/apache containers
  - mongo + nodes containers



### Docker Networks: CLI Management of Virtual Networks

~~~shell
# show networks
$ docker network ls

# Inspect a network
$ docker network inspect bridge

# Create a network
$ docker network create <NETWORK NAME>

# Run a container on a chosen network
$ docker container run -d --name proxy --newtork <NETWORK NAME> nginx

# Attach a network to container
# 컨테이너는 하나 이상의 네트워크에 붙을수 있다.
$ docker network connect <NETWORK NAME> <CONTAINER NAME>

# Detach a network from container
$ docker network disconnect <NETWORK NAME> <CONTAINER NAME>
~~~



### Docker Networks: DNS and How Containers Find Each Other

- Understand how DNS is the key to each inter-container communications
- See how it works by default with custom networks
- Learn how to use —link to enable DNS on default bridge network
- Static IP's and usign IP's for talking to containers is an anti-pattern. 
- Docker daemon has a built-in DNS server that containers user by default
- Docker defaults the hostname to the container's name, but you can also set aliases
- 같은 네트워크에 있는 컨테이너들은 컨테이너 이름으로 서로를 찾을 수 있다.   
- bridge 서버는 기본으로 DNS 서버를 탑재하고 있지 않다. 브릿지에서 작동하는 컨테이너 끼리 연결 할 때는 —link를 사용해서 서로 연결 해줘야 한다. 새로운 네트워크 만들어서 앱을 돌리는게 편하다.



~~~shell
# nginx1과 nginx2가 같은 네트워크에 있다면 서로 이름으로 ping(통신) 보낼 수 있다.
$ docker container exec -it nginx1 ping nginx2
~~~



### Assignment 1: Using Containers for CLI Testing

1. User different Linux distro containers to check **curl ** cli tool version

2. Use two different terminal windows to start bash in both centos:7 and ubuntu:14.04 using -it
3. Learn the docker container --rm option so you can save cleanup
4. Ensure curl is installed and on latest version for that distro
   - ubuntu: apt-get update && apt-get install curl
   - centos: yum update curl
5. Check curl --version



Answers

- centos:7

~~~shell
# -rm option removes the container when existing
$ docker container run --rm -it centos:7 bash

# inside the centos7 container
$ yum update curl
$ curl --version
~~~

- ubuntu:14.04

~~~shell
$ docker container run --rm -it ubuntu:14.04 bash

# inside the ubuntu:14.04 container
$ apt-get update && apt-get install curl
$ curl --version
~~~



### Assignment 2: DNS Round-Robin Test

Round-robin DNS is a technique of load balancing.

1. Ever since Docker Engine 1.11, we can have multiple containers on a created network respond to the same DNS address
2. Create a new virtual network (default bridge driver)
3. Create two containers from elasticsearch:2 image
4. Research and user **--net-alias search** when creating them to give them an additional DNS name to respond to
   - **--net-alias search** is an option for run command
5. Run **alpine nslookup search** with **--net** to see the two containers list for the same DNS name
6. Run **centos curl -s search:9200** with **--net ** multiple times until you see both "name" fields show

~~~shell
$ docker network create dude

# we can skip -p because we are not publishing to the outside interface of our host
$ docker container run -d --net dude --net-alias search elasticsearch:2
$ docker container run -d --net dude --net-alias search elasticsearch:2

# list running containers
# those ports are exposed in the virtual network
$ docker container ls 

# list servers with domain name "search"
# nslookup은 DNS 서버에 해당 도메인을 가지고 있는 서버들의 정보를 알려준다.ㅣ
$ docker container run --rm --net dude alpine nslookup search

# show a response from either of two servers
# 여러번 실행 해보자. 매번 랜덤하게 elasticsearch 서버에서 오는 것을 볼 수 있다.
$ docker container run --rm --net dude centos curl -s search:9200

# remove servers
$ docker container rm -f <container name 1> <container name 2>
~~~



# Container Images

###  What's in An Image

- App binaries and dependencies 
- Metadata about the image data and how to run the image
- Official definition: "An image is an ordered collection of root filesystem changes and the corresponding execution parameters for use within a container runtime"
- Not a complete OS. No kernel, kernel modules (e.g. drivers). Host OS provides kernels. Not booting up full operating system. It is just starting a process (application)
- Small as one file (your app binary) like a golang static binary
- Can be big as a Ubuntu distro with apt, and Apache, PHP, and more installed

~~~shell
$ docker image ls
$ docker pull nginx
$ docker pull nginx:1.11.9
$ docker pull nginx:1.11


~~~

### Images and Their Layers: Discover the Image Cache

- Images are made up of files system changes and metadata about the changes.
- Each layer is uniquely identified as a sha and only stored once on a host
- This saves storage space on host and transfer time on push/pull
- A container is just a single read/write layer on top of image

~~~shell
$ docker image history nginx:latest
$ docker image inspect nginx
~~~



###Image Tagging and Pusing to Docker Hub

Tags are just labels pointing to an image.

~~~shell
$ docker image tag --help

$ docker image ls

# this will not download new image because it is same with latest image. same image ID.
$ docker pull nginx/mainline

$ docker image tag nginx sanghapark/nginx

# new image with the same image ID of nginx.
$ docker image ls

$ docker login

$ docker image push sanghapark/nginx

$ docker image tag sanghapark/nginx sanghapark/nginx:testing
# this will not upload full new image. Awesome saving!
# if layers already exist, it won't bother downloading/uploading
$ docker image push sanghapark/nginx:testing


$ docker logout
~~~



### Building Images: Dockerfile Basics

- **FROM**

  - all Images must have a FROM
  - usually from a minimal Linux distribution like debian or alpine

  ~~~dockerfile
  FROM debian:jessie
  ~~~

- **ENV**

  - enviroment variables 

  ~~~dockerfile
  ENV NGINX_VERSION 1.11.10-1~jessie
  ~~~

- **RUN**

  - run shell command
  - all commands with && will fit into one single layer

  ~~~dockerfile
  RUN apt-get update \
  	&& apt-get install yum
  ~~~

- **EXPOSE**

  - expose these ports on the docker virtual network
  - need to use -p to open/forward these ports on host

  ~~~dockerfile
  EXPOSE 80 443
  ~~~

- **CMD**

  - required
  - run this command when container is launched

  ~~~dockerfile
  CMD ["nginx", "-g", "daemon off;"]
  ~~~



### Building Images: Running Docker Builds

~~~dockerfile
FROM debian:jessie
ENV NGINX_VERSION 1.11.10-1~jessie
RUN apt-get update \
	&& apt-get install yum
EXPOSE 80 443
CMD ["nginx", "-g", "daemon off;"]
~~~

Dockerfile will pull debian:jessie image from Docker Hub down to my local cache and executes line by line and cache each of these layers.

Let's build an image by Dockerfile

~~~shell
# dot implies to create an image on the current directory
$ docker image build -t customnginx .
~~~



### Building Images: Extending Official Images

~~~dockerfile
FROM nginx:latest

# cd to the path
WORKDIR /usr/share/nginx/html

# copy index.html inside the container
COPY index.html index.html 
~~~

Run the command below and browse localhost:80. It show default nginx default page.

~~~shell
$ docker container run -p 80:80 --rm nginx
~~~

Run Dockerfile with extra commands

~~~shell
$ docker image build -t nginx-with-html .
$ docker container run --rm -p 80:80 nginx-with-html
~~~

Browse localhost:80 and see the custom landing page.



# Container Lifetime & Persistent Data Volumes

### Data Volumes

- **VOLUME** command in Dockerfile

~~~dockerfile
# create new volume location and assign it to this directory in the container
# which means any files we put in there in the container outlive the container until
# we manually delete files
VOLUME /var/lib/mysql
~~~



### Bind Mounting

- maps a host file or directory to a container file or directory
- can't use in Dockerfile, must be at continer run

~~~shell
$ docker container run -v /Users/shp/stuff:/path/container ...
~~~





# Docker Compose, The Multi-Container Tool

- Configure relationships between containers
- Save out docker container run settings in easy-to-read file
- Create one-liner developer enviroment startups
- YAML-formatted file that describes 
  - containers
  - networks
  - volumes

- CLI tool **docker-compose** used for local dev/test automation with those YAML files

~~~yaml
version: '3.1' # if no version is specified then v1 is assumed. Recommend v2 minimum

services: # containers. same as docker run
	servicename: # a friendly name. this is algo DNS name inside network (like --name)
		image: # optional, if you use build
		command: # optional, replace the default CMD specified by the image
		environment: # optional, same as -e in docker run 
		volumes: # optional, same as -v in docker run
	sevicename2:
	
volumes: # optional, same as docker volume create

networks: # optional, same as docker network create
~~~



Example 1

~~~yaml
version: '2'

# same as
# docker run -p 80:4000 -v $(pwd):/site sanghapark/jekyll-serve

services:
	jekyll:
		image: sanghapark/jekyll-serve
		volumes:
			- .:/site
		ports:
			- '80:4000'
~~~



Example 2

~~~yaml
version: '2'

services:
	wordpress:
		image: workdpress
		ports:
			- 8080:80
		enviroment:
			WORDPRESS_DB_PASSWORD: example
		volumes:
			- ./wordpress-data:/var/www/html
			
	mysql:
		image: mariadb
		enviroment:
			MYSQL_ROOT_PASSWORD: example
		volumes:
			- ./mysql-data:/var/lib/mysql
~~~



### Basic Compose Commands

~~~shell
# setup volumes/networks and start all containers
$ docker-compose up

# stop all containers and remove cont/vol/net
$ docker-compose down
~~~

- add -d option to run in background



### Add Image Building to Compose File

~~~yaml
version: '2'

services:
	proxy: # name of running container
		build:
			context: . # build to current directory
			dockerfile: nginx.Dockerfile # nginx.Dockerfile in current directory
		image: nginx-custom # name of image
		ports:
			- '80:80'
	web:
		image: httpd
		volumes:
			- ./html:/usr/local/apache2/htdocs/
~~~



nginx.Dockerfile

~~~dockerfile
FROM nginx:1.11

COPY nginx.conf /etc/nginx/conf.d/default.conf
~~~

























