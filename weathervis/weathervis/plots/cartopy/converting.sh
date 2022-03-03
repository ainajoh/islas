#!/bin/bash
#HS; 2021-03-29
# convert images in a given folder $1 and transfer to provided date $2

here=$( pwd )

echo "converting files in $1 and transfering to $2"

# convert to smaller image size and transfer to web disk
cd $1
mkdir -p /home/centos/www/gfx/$2
lst=$( ls *.png )
echo $lst
for f in $lst; do 
  echo "converting $f"
  convert -scale 35% $f /home/centos/www/gfx/$2/$f
  \rm $f
done

# fix permissions for web access
sudo chown -R centos:apache /home/centos/www/gfx/$2

# transfer to webserver
if [[ "$HOSTNAME" == *"islas-operational.novalocal"* ]]; then
  #scp -r -i /home/centos/.ssh/islas-key.pem /home/centos/www/gfx/$2 158.39.201.233:/home/centos/www/gfx
  rsync -am --stats -r -e "ssh -i /home/centos/.ssh/islas-key.pem" /home/centos/www/gfx/$2 158.39.201.233:/home/centos/www/gfx
fi

# return to original folder
cd $here

# fin
