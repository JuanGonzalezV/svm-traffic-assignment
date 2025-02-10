#!/bin/bash
clear 

BASEDIR=$(dirname "$0")

echo "Creating Vehicular Traffic Simulation using SUMO..."
echo -n "Enter the Number of vehicles by km2 : "
read CARS

#export SUMO_HOME=/opt/sumo-src-1.2.0/sumo-1.2.0
#export SUMO_HOME=/usr/share/sumo if not already exporte in bashcr
echo -n "Enter the Area of the ROI : "
read AREA

#estos son los archivos q va a leer
DENSITY=$((CARS*AREA))
cp map.osm machala.osm
netconvert --osm-files machala.osm -o machala.net.xml
polyconvert --net-file machala.net.xml --osm-files machala.osm --type-file typemap.xml -o machala.poly.xml  


#python /opt/sumo-1.2.0/tools/randomTrips.py  -n erlangen.net.xml -r erlangen.rou.xml  -e $DENSITY -l  --intermediate 100 --trip-attributes "departLane=\"best\" departSpeed=\"max\" departPos=\"random\"" --additional-file type.add.xml --trip-attributes="type=\"hSlowCar\"" --additional-file type.add.xml

python3 randomTrips2.py  -n machala.net.xml -r machala.rou.xml -e $DENSITY -l 

sed -i 's/depart=".*"/depart="0.00"'/g $BASEDIR/machala.rou.xml #replace departure time by 0.00
echo "Creating Vehicular Traffic Simulation using SUMO..."


read -p "Do you want to copy the files to the VEINS directory in Omnet++ ? <1/0> <y/N> " prompt
if [[ $prompt == "y" || $prompt == "Y" || $prompt == "yes" || $prompt == "Yes" ]]
then 
	cp machala.rou.xml /opt/omnetpp-5.5.1/samples/veins/veins/machala.rou.xml
	cp machala.net.xml /opt/omnetpp-5.5.1/samples/veins/veins/machala.net.xml
	cp machala.poly.xml /opt/omnetpp-5.5.1/samples/veins/veins/machala.poly.xml
	echo "Please, review the values of *.playgroundSize in Omnet.ini file according to the size of the new map"

else
	#/opt/sumo-src-1.2.0/sumo-1.2.0/bin/sumo-guiD machala.sumo.cfg
	#sumo-gui -c machala.sumo.cfg
	echo "EXECUTING runner.py"
	python3 runnerMachala2.py
	exit 0
fi