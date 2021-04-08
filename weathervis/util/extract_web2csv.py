def Extract_from_webpage(file="/home/centos/progs/islas/weathervis/weathervis/data/sites.csv", url='http://www.science.smith.edu/cmet/20180921_ARC_02/flight_data.txt'):
    '''
    Get the latest position of the CMET and update the sites.csv file '''

    cr = pd.read_csv(url,sep='\s+',names=['T1','lat','lon','Unknown','T2','T3','RH'])

    # now get latest pair of lon/lat that is not NaN
    cr2 = cr.copy()
    cr2 = cr2[(cr2['lon'].notnull()) & (cr2['lat'].notnull())]

    lastlon = np.array(cr2['lon'].tail(1))[0]
    lastlat = np.array(cr2['lat'].tail(1))[0]

    # now go to sites.csv file and modify the pcmet2 point
    read_file_name = file + '.old'
    searchExp = 'pcmet3'
    replaceExp = 'pcmet3;'+str(lastlat)+';'+str(lastlon)+';None;None;None;None'

    # find line number, delete it, append new location

    os.rename(file, read_file_name)

    with open(read_file_name, 'r') as read_file:
        with open(file, 'a') as write_file:

            for n, line in enumerate(read_file, 1):
                if searchExp in line:
                    write_file.write(replaceExp)
                    print('Repleaced line no', n)
                else:
                    write_file.write(line)
    os.remove(read_file_name)



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--url",default='http://www.science.smith.edu/cmet/20180921_ARC_02/flight_data.txt', help="MEPS or")
  parser.add_argument("--file",default="/home/centos/progs/islas/weathervis/weathervis/data/sites.csv", help="MEPS or AromeArctic")
  args = parser.parse_args()
  Extract_from_webpage(file=args.file, url=args.url)
