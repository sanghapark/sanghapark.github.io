# Git을 통한 프로젝트 repository 관리

## 브랜치 종류

1. master
2. develop
3. feature
4. release
5. hotfix



1. master

   - 배포 이력을 관리하기 위해 사용 (배포 가능한 상태로만 유지)

     

2. develop

   - 다음 출시 버전을 개발하는 브랜치
   - 기능 개발 브랜치들을 병합하는 브랜치
   - 기능이 추가되고 테스트가 통과되어 배포 가능한 상태면 develop을 master에 merge한다.  

   ```shell
   # master에서 develop 브랜치 생성
   $ git checkout -b develop
   ```

   

3. feature

   - 새로운 기능 개발 및 버그 수정이 필요 할 때 develop으로부터 분기한다.
   - 기능 개발이 완료되면 develop 브랜치로 merge하여 공유

   ```shell
   # develop 브랜치에서 새로운 기능에 대한 feature 브랜치를 분기한다.
   $ git checkout -b feature/<기능 이름> develop
   
   # 새로운 기능 개발 시작...
   
   # 작업이 끝나면 develop 브랜치로 merge 한다.
   $ git checkout develop
   $ git merge --no-ff feature/<기능 이름>
   
   # 더 이상 필요하지 않은 feature 브랜치는 삭제한다.
   $ git branch -d feature/<기능 이름>
   
   # 새로운 기능이 추가된 develop 브랜치를 원격 저장소에 올린다. (push)
   $ git push origin develop
   ```

   - --no-off 옵션은 feature 브랜치에 존재하는 모든 커밋이력들을 하나의 새로운 커밋으로 만들어서 merge 하는 방식



4. release

   - 이번 출시 버전을 준비하는 브랜치
   - 배포를 귀한 전용 브랜치로써 해당 배포를 준비하는 동안 다른 팀은 다음 배포를 위한 기능을 develop에서 계속 개발 할 수 있다.
   - develop에서 배포할 수 있는 수준의 기능이 모이면 release 브랜치를 분기한다.
     - release를 만드는 순간부터 배포 사이클 시작
     - release에서는 배포를 위한 최종적인 버그수정, 문서추가 등 배포와 직접적으로 관련된 작업만 수행
   - release에서 배포 가능한 상태(모든 기능이 정상적으로 동작)가 되면
     - master에 merge 한다.
     - 배포 준비 하면서 release 브랜치에서 발생한 변경사항들을 배포 완료후에 develop에도 merge
   - release 브랜치 이름
     - ex) release-1.3.0

   ```shell
   # release-1.3.0 버전 배포 브랜치 생성
   $ git checkout -b release-1.3.0 develop
   
   #  배포 사이클 시작..
   
   # 배포를 위한 최종작업이 끝나면 master에 merge
   $ git checkout master
   $ git merge --no-ff release-1.3.0
   
   # 버전 태그 생성
   $ git tag -a 1.3.0
   
   # release에서 발생한 변경사항들 develop에 merge
   $ git checkout develop
   $ git merge --no-ff release-1.3.0
   
   # release 삭제
   $ git branch -d release-1.3.0
   ```



5. hotfix

   - 배포된 버전에서 발생한 버그를 빠르게 수정 하는경우 master에서 분기하는 브랜치
   - develop 브랜치에서 문제가 되는 부분을 수정하여 배포 가능한 버전을 만들기에는 시간이 소유 될 수 있고 안정성이 보장이 안되기 때문에 master에서 브랜치를 만들어 필요한 부분만 수정한 후 master에 merge 하여 배포한다.

   ```shell
   # master 브랜치에서 hotfix 브랜치로 분기한다.
   $ git checkout -b hotfix-1.3.1 master
   
   # 문제가 되는 부분만 빠르게 수정
   
   # master 브랜치로 이동
   $ git checkout master
   
   # master 브랜치에 hotfix-1.3.1 브랜치 내용을 merge
   $ git merge --no-ff hotfix-1.3.1
   
   # 병합한 커밋에 새로운 버전 이름으로 태그를 부여한다.
   $ git tag -a 1.3.1
   
   # develop 으로 이동
   $ git checkout develop
   
   # develop에 hotfix-1.2.1 브랜치 내용을 merge
   $ git merge --no-ff hotfix-1.3.1
   ```

   