sort -T sort_temp_dir --parallel 16 -t "," -k 404,404V -k 405,405g -k 403,403V -k 406,406V -k 407,407V Training_Data.csv | uniq > gemBS.Methylation.Training_Data.csv

# Debugging commands
awk -F $"," '{print $403,$407}' gemBS.Methylation.Training_Data.csv | paste - - - - - - - - - | sort | uniq # This should find genes where the 9 assays aren't in order

awk -F "," '{if($407>prev){if($406!=prev_strand){print "strand_mismatch",$403,$404,$405,$406,$407}}if($407==prev){print $403,$404,$405,$406,$407}prev_strand=$406;prev=$407}' gemBS.Methylation.Training_Data.csv | le # This should find genes with out of order assay labels
