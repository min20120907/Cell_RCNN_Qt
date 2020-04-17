dir1 = getDirectory("Choose source Directory");
dir2 = getDirectory("Choose destination directory");
list = getFileList(dir1);
setBatchMode(true);
for (i =0;i<list.length;i++){
	showProgress(i+1, list.length);
	
	open(dir1+list[i]);
run("Hyperstack to Stack");
run("Stack to Images");
for(j=72;j>=0;j--){
	if(j>10){
		selectWindow("new-00"+j);
	}else{
		selectWindow("new-000"+j);
	}
	saveAs("JPEG", dir2+list[i]+j);
}
	
	run("Close All");
}