import hashlib


def csv_bytes(frame):
    return frame.to_csv(index=False, date_format='%Y-%m-%d %H:%M:%S', lineterminator='\n').encode('utf-8')


def chronological_split(frame):
    if frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError('Unique valid timestamps required')
    ordered = frame.sort_values('date', kind='stable').reset_index(drop=True)
    n = len(ordered)
    sizes = [n * 70 // 100, n * 10 // 100, n * 5 // 100]
    sizes.append(n - sum(sizes))
    if min(sizes) < 1:
        raise ValueError('Insufficient rows for four nonempty partitions')
    partitions, metadata = {}, {}
    start = 0
    previous_end = None
    for name, size in zip(['train', 'validation', 'calibration', 'test'], sizes):
        part = ordered.iloc[start:start + size].copy()
        if previous_end is not None and previous_end >= part.date.iloc[0]:
            raise ValueError('Partitions overlap in time')
        partitions[name] = part
        metadata[name] = dict(rows=size, start=str(part.date.iloc[0]), end=str(part.date.iloc[-1]),
                              start_index=start, stop_index_exclusive=start + size,
                              sha256=hashlib.sha256(csv_bytes(part)).hexdigest())
        previous_end = part.date.iloc[-1]
        start += size
    return partitions, dict(rule='floor 70%,10%,5%; all remainder to Test',
                            nonoverlap_verified=True, total_rows=n, partitions=metadata)
