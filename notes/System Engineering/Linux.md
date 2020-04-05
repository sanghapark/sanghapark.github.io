# Linux

### 계정

- 계정이 있는지 확인

  ~~~shell
  $ cat /etc/passwd | grep sanghapark
  ~~~

  결과 없으면 sanghapark 계정 없음

- 계정 생성

  ~~~shell
  $ useradd sanghapark -m -s /bin/bash # Ubuntu
  $ useradd sanghapark # CentOS
  ~~~

  -m 옵션을 명시해야 홈 디렉토리가 생성됨

  -s /bin/bash 옵션을 명시해야 쉘 환경이 설정됨

- 패스워드 

  ~~~shell
  $ sudo passwd sanghapark
  ~~~

  새로운 비밀번호를 입력하면 된다.

- 계정 변경

  ~~~shell
  sudo sanghapark
  ~~~

  sanghapark 계정으로 변경한다. 비밀번호가 있다면 비밀번호를 입력해야 한다.

- 