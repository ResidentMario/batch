if __name__ == "__main__":
    from distributed import Client, LocalCluster
    import dask.dataframe as df
    import dask.array as da

    cluster = LocalCluster()
    client = Client(cluster)

    matches = da.from_npy_stack("data/")
    matches = df.from_array(matches)

    # IMPORTANT: note that this repartition is optional, if you want a partitioned write
    matches.repartition(npartitions=1).to_csv("predictions.csv")
