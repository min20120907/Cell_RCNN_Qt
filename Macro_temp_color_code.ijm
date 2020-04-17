dir1 = getDirectory("Choose source Directory");
dir2 = getDirectory("choose destination directory");
list = getFileList(dir1);
setBatchMode(true);
for (i =0;i<list.length;i++){
	showProgress(i+1, list.length);
	
	open(dir1+list[i]);
	run("Split Channels");
	run("Enhance Contrast", "saturated=0.35");
	run("Enhance Contrast", "saturated=0.35");
	run("Temporal-Color Code", "lut=Fire start=1 end=73 create");
	selectWindow("MAX_colored");
	saveAs("PNG", dir2+list[i]);
	run("Close All");
}
