run("Merge Channels...", "c1=C1-s1 c2=C2-s1 create");
run("RGB Color", "slices keep");
run("Temporal-Color Code", "lut=Fire start=1 end=40 create");
