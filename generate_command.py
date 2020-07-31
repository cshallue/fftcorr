def command(NGC, num):
    com = "./fftcorr -cell 3.0 -sep 300.0 -dsep 1.0 -ell 4 -k 0.4 -dk 0.005 "
    if (NGC=='N'):
	com = com + "-n3 1024 1024 750 "
    else:
	com = com + "-n3 750 750 625 "
    if (num>0):
	in1  = "-in Patchy/binary/patchy-DR12CMASS-%s-V6C-%04d.dat " % (NGC, num)
	in2 = "-in2 Patchy/binary/patchy-DR12CMASS-%s-V6C-random50.dat " % (NGC)
	out = "-out Patchy/Output/patchy-DR12CMASS-%s-V6C-%04d-c3r300.dat " % (NGC, num)
	com = com + in1 +in2 + out
    else:
	in1 = "-in Patchy/binary/patchy-DR12CMASS-%s-V6C-random50.dat " % (NGC)
	out = "-out Patchy/Output/patchy-DR12CMASS-%s-V6C-ran-c3r300.dat " % (NGC)
	com = com + in1 + out
    print "echo ", com
    print "time ", com

def patchy(min, max):
    for num in range(min, max+1):
	command('N', num)
	command('S', num)

patchy(0,0)
patchy(1,9)     # Skip number 10, which seems to be missing
patchy(11,600)
