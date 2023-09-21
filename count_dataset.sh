#!/bin/bash
for prefix in nuclear chromosome data; do
  echo "$prefix: $(sort via_region_$prefix*.json | grep -o "\"filename\": [^:]*" | uniq -c | wc -l) images"
done
