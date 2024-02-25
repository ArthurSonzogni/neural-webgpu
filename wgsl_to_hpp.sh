#!/bin/sh

filename=$1
wgsl_file=$2
hpp_file=$3

echo "Generating:"
echo "filename = $filename"
echo "wgsl_file = $wgsl_file"
echo "hpp_file = $hpp_file"


mkdir -p $(dirname $hpp_file)

echo "namespace wgsl {" > $hpp_file
echo "  constexpr const char* ${filename} = R\"wgsl("  >> $hpp_file
cat $wgsl_file \
  | sed 's/{/{{/g' \
  | sed 's/}/}}/g' \
  | sed 's/{{}}/{}/g' \
  >> $hpp_file
echo ")wgsl\";" >> $hpp_file
echo "} // namespace wgsl" >> $hpp_file
