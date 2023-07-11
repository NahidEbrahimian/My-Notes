# Git Commands

### Rename Folder

```
git mv old_name new_name
git status
git add *
git commit -m "your message
git push -u origin main
```
### Delete File from Repository

First clone your repository and then:

```
git rm file-name
git commit -m "your message"
git push -u origin main
```

### Tag

First push and then tag:

```
git status
git add *
git branch
git checkout branch_name
git commit -m "your message"
git push -u origin branch_name
git tag -a version_code -m "your message"
git push origin version code
```

### Delete sumbmodule from repository

First clone your repository and then:

```
git submodule deinit -f submodule_directory
rm -rf submodule_directory
git rm -f submodule_directory
```

### Create new branch

create your new branch called "subbranch_of_b1" under the "branch1":

```
git checkout branch1
git checkout -b subbranch_of_b1 branch1
```

### Merge two branches:

```
git checkout branch1
git merge subbranch_of_b1
```

### Delete a remote branch

```
# delete a local branch:
git branch -d branch-name

# removing a remote branch:
git push origin -d branch-name
```

### Undo the last commit

```
git reset --soft HEAD~1
```
### Remove Commit

In local:

using this command, get your commit code that you want to remove:

```
git log
```

then, 

```
git revert commit-code
```

### Remove file or directory from repository:

```
git rm -r --cached ./runs/train/
git commit -m "[remove]remove /train/runs directory"
git push -u origin yolov7-face
```
