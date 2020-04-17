/*
 *操作方式：
 *1.點選Run 2.選擇輸入檔案 3.選擇輸出位置 4.填入ROI寬度 5.等待輸出

*/
//main
if(roiManager("Count")>0){
	roiManager("Deselect");
	roiManager("Delete");
}
run("Close All");
inp=File.openDialog("Select a file");
roiManager("Open", File.openDialog("Select a ROI file"));
outp=getDirectory("Choose a output Directory");
str=Name(inp);
open(inp);
N_frames=nSlices/2;
title=getTitle();
//
run("Split Channels");
selectWindow("C1-"+title);
resetMinAndMax();
run("Enhance Contrast", "saturated=0.35");
run("Apply LUT", "stack");
selectWindow("C2-"+title);
resetMinAndMax();
run("Enhance Contrast", "saturated=0.35");
run("Apply LUT", "stack");
run("Concatenate...", " image1=C1-"+title+" image2=C2-"+title);
saveAs("Tiff", outp + str +  " 0.tif");
//
for (i = 0; i < 8; i++)File.makeDirectory(outp +"S"+(i+1)+"");
inp=outp + str + " 0.tif";
str=str+" ";
output=outp+"S1";
Fun_1(inp,output,str);
input=outp+"S1";
output=outp+"S2";
Fun_2(input,output);
input=outp+"S2";
output=outp+"S3";
Fun_3(input,output);
input=outp+"S3";
output=outp+"S4";
Fun_4(input,output,N_frames);
input=outp+"S1";
output=outp+"S5";
Fun_5(input,output,1);
input=outp+"S1";
output=outp+"S6";
Fun_5(input,output,2);
run("Close All");
input=outp+"S"+5+"";
output=outp+"S7";
Fun_7(input,output,N_frames);
run("Close All");
input=outp+"S7";
inputB=outp+"S4";
output=outp+"S8";
Fun_8(input,inputB,output);
run("Close All");
print("All Finished");

//Function 1
function Fun_1(input,output,str){
	open(input);
	filename=getTitle();
	width=getNumber("Max width of ROI",0);
	for (i=0; i<roiManager("count");++i){
    	roiManager("Select", i);
    	run("Straighten...", "title= line="+width+" process");
    	run("Stack to Hyperstack...", "order=xytcz channels=2 slices=1 frames="+nSlices/2+" display=Color");
		saveAs("Tiff", output + str + (i+1) + ".tif");
		close();
    	selectWindow(filename);
	}
	run("Close All");
}

//Function 2
function Fun_2(dir,output){	//processFolder_2
	list = getFileList(dir);
	for (i=0; i<list.length; i++) {
		if(endsWith(list[i], ".tif"))processFile_2(dir, output, list[i]);  //add the file ending for your images
		else if(endsWith(list[i], "/") && !matches(output, ".*" + substring(list[i], 0, lengthOf(list[i])-1) + ".*"))
			processFolder_2(""+dir+list[i]);//if the file encountered is a subfolder, go inside and run the whole process in the subfolder
		else print(dir + list[i]); //if the file encountered is not an image nor a folder just print the name in the log window
	}
}

function processFile_2(inputFolder, output, file){
	open(inputFolder + file);
	title = getTitle();
	run("Split Channels");
	two = "C2-" + title;
	one = "C1-" + title;
	selectWindow("C1-"+title);
	selectWindow("C2-"+title);
	run("Merge Channels...", "c1=["+one+"] c2=["+two+"] ");
	saveAs("tiff", output + file);
	run("Close All");
}

//Function 3
function Fun_3(dir,output) {	//processFolder_3
	list = getFileList(dir);
	for (i=0; i<list.length; i++) {
		if(endsWith(list[i], ".tif"))processFile_3(dir, output, list[i]);  //add the file ending for your images
		else if(endsWith(list[i], "/") && !matches(output, ".*" + substring(list[i], 0, lengthOf(list[i])-1) + ".*"))
			processFile_3(""+dir+list[i]);
   		    //if the file encountered is a subfolder, go inside and run the whole process in the subfolder
		else print(dir + list[i]);	//if the file encountered is not an image nor a folder just print the name in the log window
    }
    run("Close All");
}

function processFile_3(inputFolder, output, file) {
	open(inputFolder + file);
	title = getTitle();
	run("Make Montage...", "columns=1 rows="+nSlices+" scale=1");
	resetMinAndMax();
	run("8-bit");
	//setTool("zoom");
	setAutoThreshold("Default dark");
	//run("Threshold...");
	//setThreshold(15, 255);
	setOption("BlackBackground", false);
	run("Convert to Mask");
	saveAs("tiff", output +"cellmask-montage "+ file);
	run("Close All");
}

//Function 4
function Fun_4(dir,output,N_frames) {	//processFolder_4
	list = getFileList(dir);
	for (i=0; i<list.length; i++) {
		if(endsWith(list[i], ".tif"))processFile_4(dir, output, list[i],N_frames);	//add the file ending for your images
		else if(endsWith(list[i], "/") && !matches(output, ".*" + substring(list[i], 0, lengthOf(list[i])-1) + ".*"))
			Fun_4(""+dir+list[i],output,N_frames);	//if the file encountered is a subfolder, go inside and run the whole process in the subfolder	    
		else print(dir + list[i]);	//if the file encountered is not an image nor a folder just print the name in the log window
	}
	run("Close All");
}

function processFile_4(inputFolder, output, file, N_frames) {
	open(inputFolder + file);
	title = getTitle();
	run("Montage to Stack...", "images_per_row=1 images_per_column="+N_frames+" border=0");
	run("Fill Holes", "stack");
	saveAs("tiff", output +"cellmask-stack_"+ file);
	run("Close All");
}

//Function 5-6
function Fun_5(dir,output,channel){	//processFolder_5
	list = getFileList(dir);
	for (i=0; i<list.length; i++){
		if(endsWith(list[i], ".tif"))processFile_5(dir, output, list[i],channel); //add the file ending for your images
		else if(endsWith(list[i], "/") && !matches(output, ".*" + substring(list[i], 0, lengthOf(list[i])-1) + ".*"))
			processFolder(""+dir+list[i],channel);	//if the file encountered is a subfolder, go inside and run the whole process in the subfolder 
		else print(dir + list[i]);	//if the file encountered is not an image nor a folder just print the name in the log window
    }
}
function processFile_5(inputFolder, output, file, channel) {
	open(inputFolder + file);
	title = getTitle();
	run("Split Channels");
	two = "C2-" +title;
	one = "C1-" +title;
	selectWindow("C"+channel+"-"+title);
	saveAs("tiff", output + "C"+channel+" " + file);
	run("Close All");
}

//Function 7
function Fun_7(dir,output,N_frames){
	list = getFileList(dir);
	for (i=0; i<list.length; i++) {
		if(endsWith(list[i], ".tif"))MakeSeg(dir, list[i], output, N_frames);  //
		else if(endsWith(list[i], "/") && !matches(output, ".*" + substring(list[i], 0, lengthOf(list[i])-1) + ".*"))
			MakeSeg(dir,""+dir+list[i],output, N_frames);
		else print(dir + list[i]);
	}
}
function MakeSeg(input_dir, file, output, N_frames){
	open(input_dir+file);
	title = getTitle();
	run("Make Montage...","columns=1 rows="+N_frames+" scale=1");
	selectWindow("Montage");
	run("Gaussian Blur...","sigma=30");
	run("Find Maxima...","noise=100 output=[Segmented Particles]");
	selectWindow("Montage Segmented");
	run("Montage to Stack...", "columns=1 rows="+N_frames+" border=0");
	saveAs("tiff", output +"Segmented Particles "+ file);
	run("Close All");
}

//Function 8
function Fun_8(input_FolderA,input_FolderB,OutputFolder){//Input folder
	listA = getFileList(input_FolderA);
	listB = getFileList(input_FolderB);
	for(i=0;i<listA.length;i++)CellCalculator(input_FolderA+listA[i],input_FolderB+listB[i],OutputFolder,i);
}
function CellCalculator(inputA,inputB,output,outNum){ // Input file,output folder
	open(inputA);
	titleA = getTitle();
	open(inputB);
	titleB = getTitle();
	imageCalculator("AND create stack", titleA , titleB);
	//selectWindow("Result of Untitled");
	//rename("ImageCalculator_");
	SaseNum=outNum+1;
	saveAs("tiff", output +"Cell Calculator "+ SaseNum+".tiff");
	run("Close All");
}
function Name(path){
	p=path;
	if(lastIndexOf(p,"")== lengthOf(p)){
		s1=lastIndexOf(p,"");
		p=substring(p,0,s1);
		s1=lastIndexOf(p,"");
	}
	else s1= lastIndexOf(p,"");
	str=substring(p,s1+1,lengthOf(p));
	str=removeExtension(str,"tif");
	return str;
}
function removeExtension(path,Extension){
	S_extension="."+Extension;
	s= lastIndexOf(path, S_extension);
	if(s==-1)return path;
	else return substring(path,0,s);
}