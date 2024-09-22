PROCESS=`ps -ef | grep -v grep | grep main.py | grep webserver | awk '{print $2}'`
for i in ${PROCESS}
do
  kill -9 $i
  echo "killed $i"
  sleep 1
done

sleep 1

nohup ./run.sh webserver > user_data/logs/webserver.log 2>&1 &
