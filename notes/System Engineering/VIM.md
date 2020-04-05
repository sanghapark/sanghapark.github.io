https://gist.github.com/pythoninthegrass/8471bed151c38edad84c73b3e6c617d1



```shell
$ vim --version

$ sudo yum install -y python-devel

$ sudo yum install -y vim

$ vim --version

$ git clone https://github.com/vim/vim.git

$ cd vim

$ ./configure --with-features=huge --enable-gui=no --without-x --enable-perlinterp --enable-pythoninterp --enable-tclinterp --enable-rubyinterp --with-python-config-dir

$ make

$ sudo make isntall

$ sudo cp -ru ~/vim/src/vim /usr/bin 
$ cp -ru ~/vim/src/vim /usr/local/bin

$ sudo yum update vim*

$ vim --version

$ sudo yum install -y git

$ git clone https://github.com/gmarik/Vundle.vim.git ~/.vim/bundle/Vundle.vim
```



### Plugins to install

- surround.vim: quoting/parenthesizing made simple
- The NERD tree
- 