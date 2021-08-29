# Server

#### General system functions:

* Ping the server: https://python-binance.readthedocs.io/en/latest/general.html#id1
* Get system status: https://python-binance.readthedocs.io/en/latest/general.html#id3
* Check server time and compare with local time: https://python-binance.readthedocs.io/en/latest/general.html#id2

#### Time synchronization in OS and time zones

https://www.digitalocean.com/community/tutorials/how-to-set-up-time-synchronization-on-ubuntu-16-04

Check synchronization status:
```
$ timedatectl
```

"Network time on: yes" means synchronization is enabled. "NTP synchronized: yes" means time has been synchronized.

If timesyncd isn’t enabled, turn it on with timedatectl:
```
$ sudo timedatectl set-ntp on
```

Using chrony:

https://www.fosslinux.com/4059/how-to-sync-date-and-time-from-the-command-line-in-ubuntu.htm

```
$ sudo apt install chrony
chronyd  # One-shot time check without setting the time
chronyd -q  # One-shot sync
```

#### Create virtual environment

```
$ python3.7 -m pip install --user pip --upgrade
$ python3.7 -m pip install --user virtualenv --upgrade
```

```
$ python3.7 -m virtualenv --version
virtualenv 20.0.13
$ python3.7 -m virtualenv venv
```

#### Start from Linux

Modify start.py by entering data collection command. Alternatively, pass the desired command as an argument.
See the file for additional comments. For example:
* `collect_data` is used to collect depth data by making the corresponding requests.
  * It is possible to specify frequency 1m, 5s etc.
  * It is possible to specify depth (high depth will decrease weight of the request)
* `collect_data_ws` is used to collect stream data like klines 1m and depth.
  * klines will get update every 1 or 2 seconds for the current 1m kline
  * Depth stream will send new depth information (limited depth) every 1 second
  * Other streams could be added to the app configuration

```
switch to the project root dir
$ source venv/bin/activate OR source ../trade/venv/bin/activate
(venv) $ python3.7 --version
Python 3.7.3
(venv) $ nohup python3.7 start.py &
(venv) $ nohup python3.7 start.py collect_data_ws &
<Enter>
$ logout
```
End:
```
login
ps -ef | grep python3.7
kill pid_no
```
#### Compress and download collected data files

Zip into multiple files with low priority one file:
```
nice -n 20 zip -s 10m -7 dest.zip source.txt
```

Information about zip and nice (priority):
```
zip -s 100m archivename.zip filename1.txt
zip -s 100m -r archivename.zip my_folder  # folder and all files recursively
nice -10 perl test.pl - run with niceness 10 (lower priority) (- is hyphen - not negative).
nice --10 perl test.pl - start with high priority (negative niceness)
nice -n -5 perl test.pl - increase priority
nice -n 5 perl test.pl - decrease priority
nice -n 10 apt-get upgrade - start with lower priority (lower values of niceness mean higher priority, so we need higher values)
```

#### Sudden reboots

Information about last reboot:
```
last reboot

tail /var/log/syslog or less /var/log/syslog
```

System wide logger:
```
tail /var/log/syslog
less /var/log/syslog
```
Kernel log:
```
tail /var/log/kern.log
```

Example automatic reboot:

```
last reboot
reboot   system boot  4.15.0           Thu Apr 30 08:55   still running
reboot   system boot  4.15.0           Thu Apr 30 08:21   still running
```

```
syslog
Apr 30 06:03:01 linux CRON[23790]: (root) CMD (cd / && run-parts --report /etc/cron.hourly)
Apr 30 06:40:33 linux systemd[1]: Starting Daily apt upgrade and clean activities...
Apr 30 06:40:34 linux systemd[1]: Started Daily apt upgrade and clean activities.
Apr 30 06:55:06 linux systemd[1]: getty@tty2.service: Service has no hold-off time, scheduling restart.
Apr 30 06:55:06 linux systemd[1]: getty@tty2.service: Scheduled restart job, restart counter is at 876.
```

Check available timers, particularly, daily upgrade timer:
```
sudo systemctl list-timers
Fri 2020-05-01 06:14:53 UTC  19h left      Thu 2020-04-30 06:40:33 UTC  3h 58min ago apt-daily-upgrade.timer      apt
```
Solutions (https://superuser.com/questions/1327884/how-to-disable-daily-upgrade-and-clean-on-ubuntu-16-04):
* simply remove package unattended-upgrades: apt-get remove unattended-upgrades (but it might be insufficient)
* disable:
```
systemctl stop apt-daily-upgrade.timer
systemctl disable apt-daily-upgrade.timer
(systemctl disable apt-daily.service) - not clear if necessary
systemctl daemon-reload
```
