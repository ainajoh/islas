def Extract_from_webpage(
    file="/home/centos/progs/islas/weathervis/weathervis/data/sites.yaml",
    url="http://www.science.smith.edu/cmet/20180921_ARC_02/flight_data.txt",
):
    """
    Get the latest position of the CMET and update the sites.yaml file"""

    cr = pd.read_csv(
        url, sep="\s+", names=["T1", "lat", "lon", "Unknown", "T2", "T3", "RH"]
    )

    # now get latest pair of lon/lat that is not NaN
    cr2 = cr.copy()
    cr2 = cr2[(cr2["lon"].notnull()) & (cr2["lat"].notnull())]

    lastlon = np.array(cr2["lon"].tail(1))[0]
    lastlat = np.array(cr2["lat"].tail(1))[0]

    # read sites.yaml file
    with open(file, "r") as stream:
        try:
            sites = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.exception(exc)
            raise

    # update pcmet3 point
    sites["pcmet3"]["lat"] = lastlat
    sites["pcmet3"]["lon"] = lastlon
    sites["pcmet3"]["height"] = None

    # rewrite sites.yaml
    # Note: we lose comments on the file
    with open(file, "w") as stream:
        try:
            yaml.safe_dump(sites, stream, default_flow_style=False)
        except yaml.YAMLError as exc:
            logging.exception(exc)
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://www.science.smith.edu/cmet/20180921_ARC_02/flight_data.txt",
        help="MEPS or",
    )
    parser.add_argument(
        "--file",
        default="/home/centos/progs/islas/weathervis/weathervis/data/sites.yaml",
        help="MEPS or AromeArctic",
    )
    args = parser.parse_args()
    Extract_from_webpage(file=args.file, url=args.url)
