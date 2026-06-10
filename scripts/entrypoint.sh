#!/bin/zsh

# Start the SSH daemon in the background
/usr/sbin/sshd

# 主機檔案權限：容器以 root 執行，bind mount 內新建的檔案在主機上會變 root 擁有，
# 導致 MobaXterm/SFTP permission denied。umask 000 讓新檔案任何人可寫（容器執行中也能即時編輯），
# 編譯後與離開容器時再把擁有者還給主機使用者 (HOST_UID，預設 ubuntu=1000)。
umask 000
HOST_UID="${HOST_UID:-1000}"

# Compile the project
source /opt/ros/noetic/setup.zsh
catkin_make
chown -R "$HOST_UID:$HOST_UID" /root/catkin_ws
source /root/catkin_ws/devel/setup.zsh
cd /root/catkin_ws

# 把預先打包的 rosdebug skill 放進 Claude 設定目錄 (每次開機刷新，idempotent)。
# CLAUDE_CONFIG_DIR 由 Makefile 掛載到主機，所以 skill 與登入憑證都會落在持久目錄。
CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-/root/.claude}"
if [ -d /opt/claude-skills/rosdebug ]; then
  mkdir -p "$CLAUDE_DIR/skills"
  cp -r /opt/claude-skills/rosdebug "$CLAUDE_DIR/skills/"
  echo "Loaded rosdebug skill into $CLAUDE_DIR/skills/"
  echo " "
fi

# Setup USB connection
echo "Remap the serial port(ttyUSBX, ttyACMX) to custom name"
echo " "

echo "Rplidar usb connection as /dev/rplidar"
echo "Plate usb connection as /dev/plate"
echo "Arduino usb connection as /dev/arduino"
echo "Camera usb connection as /dev/camera"
echo "Realsense camera usb connection as /dev/realsensecamera"
echo " "

echo "Check these using the command : ls -l /dev|grep ttyUSB"
echo "Check the detail of the connection, using the command: udevadm info --attribute-walk /dev/ttyUSBX"
echo "(replace the /dev/ttyUSBX with your target device)"
echo " "

echo "Start copy rule files in scripts, to /etc/udev/rules.d/"
cp /root/scripts/rplidar.rules /etc/udev/rules.d
cp /root/scripts/plate.rules /etc/udev/rules.d
cp /root/scripts/arduino.rules /etc/udev/rules.d
cp /root/scripts/camera.rules /etc/udev/rules.d
cp /root/scripts/realsensecamera.rules /etc/udev/rules.d
echo " "

echo "Restarting udev"
service udev restart
udevadm control --reload-rules
udevadm trigger
echo " "

echo "Finish usb port setup"
echo " "

# 執行主要指令（互動 zsh）。不能用 exec，否則離開後無法做最後的權限修復。
"$@"

# 離開容器前把工作區擁有者還給主機使用者，確保主機端編輯不會 permission denied
echo "Restoring /root/catkin_ws ownership to host user (uid $HOST_UID)..."
chown -R "$HOST_UID:$HOST_UID" /root/catkin_ws /root/agent_ref 2>/dev/null
echo "Done."