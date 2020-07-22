#!/bin/bash
mkdir output
jq . via_region_data.json > output/via_region_data.json
sed -i 's/\//\\\\/g' output/via_region_data.json
