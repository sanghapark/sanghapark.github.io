# Vagrant

## Quick Installation: Mac OS X

- Download VirtualBox x.x.x platform packages and Virtual x.x.x Oracle VM Virtual Extension Pack
- Install VirtualBox
- Go to Preferences and add Extension Pack



## Vagrant Quick Start by Example

- Vagrant version

  ~~~shell
  $ vagrant version
  ~~~

- create Vagrant enviroment

  ~~~shell
  $ vagrant init hashicorp/precise64
  ~~~

  This creates Vagrantfile on the current directory

- start Vagrant enviroment

  ~~~shell
  $ vagrant up
  ~~~

  Vagrant will look for the box locally. If it cannot find locally, it will download box from online.

- Vagrant status

  ~~~shell
  $ vagrant status
  ~~~

  This shows if Vagrant is running or not

- Access the virtual machine

  ~~~shell
  $ vagrant ssh
  ~~~

  This shows the running virtual machine and makes the user logged in the virtual machine.

  In the virtual machine, /vagrant folder is the shared directory between the host and the virtual machine.

- Let the virtual machine sleep

  ~~~shell
  $ vagrant suspend
  ~~~

  We can recover the virtual machine by just typing 'vagrant up' command.

- Shut down the virtual machine

  ~~~shell
  $ vagrant halt
  ~~~

  Recover by typing 'vagrant up' command

- Destroy the virtual machine

  ~~~shell
  $ vagrant destroy
  ~~~

  If I have changed anything in the virtual machine, it will be lost.

- Halt and up the virtual machine 

  ~~~shell
  $ vagrant reload
  ~~~

- Include .vagrant in .gitignore file

## Vagrant Boxes

- Visit the following website to download boxes

  - http://www.vagrantbox.es/

  - Search opscode lentos 7.1

  - Download the box

    ~~~shell
    $ vagrant box add centos-7.0 http://opscode-vm-bento.s3.amazonaws.com/vagrant/virtualbox/opscode_centos-7.0_chef-provisionerless.box
    ~~~

- List locally available boxes

  ~~~shell
  $ vagrant box list
  ~~~

- Log into the virtual machine and type

  ~~~shell
  [vagrant@localhost]$ cd /etc
  [vagrant@localhost]$ cat system-release
  ~~~

  It show the operting system.

- Another option to find Vagrant boxes

  - https://app.vagrantup.com/boxes/search

- To preserve the changes in the virtual machine, we need to create own custom vagrant box

  ~~~shell
  $ vagrant package --ouput custom-centos-7.0.box
  ~~~

  This creates a custom-centos-7.0.box file.  This is fully boxed-up virtual machine image file that Vagrant can use. Let's add it to our system.

  ~~~shell
  $ vagrant box add Custom-Centos7 custom-centos-7.0.box
  ~~~

  Check if it is successfully added by typing 'vagrant box list'



## Vagrant Provisioning

- Update Vagrant box

  ~~~shell
  $ vagrant box update
  ~~~

  When there is a latest version update, it will show a message while Vagrant up.

- Bash Shell Provisioning

  NGINX 로 프록시 서버를 만들어보자.

  아래는 톰캣 서버 만들다가 맘.

  First, create a shell script file for provisioning like provision.sh

  ~~~shell
  #!/bin/bash
  apt-get update -y
  apt-get upgrade -y
  apt-get install -y nano git openjdk-7-sdk openjdk-7-jre-headless
  ~~~

  Open Vagrantfile and add the following

  ~~~ruby
  ...
      
  config.vm.box = "centos-7.0"
  
  # add below
  config.vm.provision "shell", path: "provision.sh"
  
  ...
  ~~~

- Rerun the provision

  ~~~shell
  $ vagrant provision
  ~~~


## Account

- 루트 계정 비밀번호 설정

  ~~~shell
  $ sudo passwd root
  ~~~

- 계정 생성

  ~~~shell
  $ sudo root # 루트 계정으로 로그인
  $ sudo adduser <username> # 계정 생성
  $ sudo passwd <username>  # 비밀번호 설정
  ~~~

- 계정에 sudo privileges 부여

  ~~~shell
  $ sudo usermod -aG wheel <username> # 계정 wheel 그룹에 추가
  ~~~

  wheel은 sudo privileges를 지닌 그룹이다.











































