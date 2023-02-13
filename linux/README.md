# Linux Commands

#### unzip .tar file

```
tar -xvf file_name.tar.gz -C path/to/directory
```

#### unzip .zip file

```
unzip file_name.zip -d path/to/directory
```

#### tmux

```
# create
tmux new -n session_name

# attach to existing session
tmux at -t session_name

# delete
tmux kill-session -t session_name
```
#### count number of file in directory

```
ls ./directory | wc -l
```
#### move files

```
mv ./source_directory ./destination_directory
```

#### internet connectivity

```
ping google.com
#or
ping 8.8.8.8
```

#### CPU, RAM, and GPU usage monitoring

```
htop
top -i
psutils
watch -n 0.5 nvidia-smi
```

#### Run .sh file shell script

```
# Set execute permission on your script using chmod command :

chmod +x script-name-here.sh

# Run your script :

./script-name-here.sh

or

sh script-name-here.sh

or

bash script-name-here.sh
```

### Create new directory

```
mkdir directory_name

or 

mkdir -p parent_directory_name/directory_name
```
