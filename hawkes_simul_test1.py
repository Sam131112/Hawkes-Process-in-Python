import numpy as np
import random as rn
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import codecs
from collections import OrderedDict
from collections import defaultdict
import scipy.stats as st
import random
import networkx as nx
import time
import sys
import difflib

mu = 0.0468
al = 36.392
bi = 133.567

i=0
times = []
s = 0
T = 1000
lmdb = 0
atimes = []
rates = []


def loglik(tm,param):
		t = np.array(tm)
		R = [0.0 for x in xrange(len(tm))]
		for i in range(len(tm)):
					try:
					        if i==0:
					                R[i] = 0.0
				        	else:
				                	k = math.exp(-param[2]*(t[i] - t[i-1]))*(1+R[i-1])
				                	R[i]=k
		            		except:
				    		#print i,param[2],t[i],t[i-1],R[i-1]
				    		R[i] = 0
	    	r = np.sum(np.log(param[1]*np.array(R)+param[0]))
		return (param[0]*tm[-1]-(param[1]/param[2])*np.sum(np.exp(-param[2]*(t[-1]-t))-1)-r)



def lambda_func(times,s,mu,al,bi):
	if s == 0:
		return mu
	elif len(times) == 0:
		return mu
	else:							
		l = mu	 + np.sum(al*np.exp(-bi*(s-np.array(times)))) 			
		return l



def hawks_estimation(event):
		#print event[:20]
		x = [3.0,3.5,4.1]
		h_est = lambda x:loglik(event,x) 
		#d =  minimize(h_est,x,method="L-BFGS-B",bounds=((0.001,None),(0.001,None),(0.009,None)),options={'disp':False,'maxiter':500})
		cons = ({'type': 'ineq','fun' : lambda x: np.array([x[2] - x[1]-1e-5])})
		#d =  minimize(h_est,x,method="SLSQP",bounds=((1e-5,None),(1e-5,None),(1e-5,None)),constraints=cons,jac=h_jac,options={'disp':False,'maxiter':1000}) 
		d =  minimize(h_est,x,method="SLSQP",bounds=((1e-5,None),(1e-5,None),(1e-5,None)),constraints=cons,options={'disp':False,'maxiter':1000}) 
		#d =  minimize(loglik,[3.0,3.5,4.1],method="L-BFGS-B",bounds=((0.001,None),(0.001,None),(0.001,None)),options={'disp':False,'maxiter':500})
		#d = minimize(loglik,[3.0,4.5,5.5],method='nelder-mead')
		#print loglik((3,4,4.5))
		#print _R[:10]
		#print "Log Likehood"
		return d.x



def ranges(c):
	i = 1
	while(i<=c):
		yield i
		i=i+1





def make_plots(i,files):
									f = codecs.open(files+"/All_Link_Inter_"+str(i)+".txt","r","utf-8")
									counter = 0
									temp = OrderedDict()
									#print "poisson"
									for x in f:
											x = x.strip("\n\r")
											x=x.split("\t")
											val = [float(y) for y in x[1:len(x)-1]]
											temp[x[0]] = val	

									'''
									for u,v in temp.items():
											if u.endswith("s"):
												k = u[0:-1]
												if k in temp:
													v.extend(temp[k])
													del temp[k]
									'''
									counts = 1
									rates = {}
									for x,x1 in temp.items():
											val = temp[x]
											if len(val)>20:
												#print x,np.mean(buck),st.expon.fit(buck,floc=0)[1],st.expon.fit(buck)[0],len(buck)
												r1 = st.expon.fit(val,floc=1)[1]
												r = np.mean(val)
												#print x,1.0/r
												rates[x] = 1.0/r
												#print x,rates[x]
									return rates
									f.close()




'''
if __name__ == '__main__':
		
		file_name = sys.argv[1]
		fno = sys.argv[4]
		c3 = pickle.load(open("pages_cat_dict.p","rb"))
		c1 = pickle.load(open("category_dic.p","rb"))
		c2 = pickle.load(open("category_dic_rev.p","rb"))
		g = nx.read_edgelist("category_graph.txt",nodetype=int)
		f = open(file_name+"/All_Link_context_"+fno+".txt","r")
		no = []
		temps = OrderedDict()
		#print "poisson"
		for x in f:
				x = x.strip("\n\r")
				x=x.split("\t")
				val = [float(y) for y in x[1:len(x)-1]]
				temps[x[0]] = val	

		
		for u,v in temps.items():
				if u.endswith("s"):
					k = u[0:-1]
					if k in temps:
						v.extend(temps[k])
						del temps[k]
		
		trivial = defaultdict(list)
		triv = set()
		flag = 0
		for u,v in temps.items():                  # Finding Trivial Pairs.
			if len(v)>20:
				for x,y in temps.items():	
					if x!=u and len(v)>20:
						sq = difflib.SequenceMatcher(None,v,y)
						if sq.ratio()>=0.75:
							trivial[u].append(x)




		print len(trivial),len(temps)
		for x in trivial:
			print x,trivial[x]
		alphas = []
		mus = []
		betas =[]
		no = [int(fno)]
		rates1 = []
		rtes = make_plots(int(fno),file_name)
		page = sys.argv[2]
		cat_nos = []
		if page in c3:
			cats = c3[page]
		for x in cats:
			if x in c1:
				cat_nos.append(c1[x])
		#print cat_nos,cat_nos[0]
		c_no = int(sys.argv[3])
		g1 = nx.bfs_tree(g,cat_nos[c_no])
		short_dict = nx.single_source_shortest_path_length(g1,cat_nos[c_no])
		for i in no:
				rates = []
				f = codecs.open(file_name+"/All_Link_"+str(i)+".txt","r","utf-8")
				temp = OrderedDict()
				temp_du = OrderedDict()
				#print "hawkes"
				for x in f:
					try:
						x = x.strip("\n\r")
						x=x.split("\t")
						val = [float(y) for y in x[1:len(x)-3]]
						date_t1 = x[-2]
						date_t2 = x[-3]
						pattern = "%d/%m/%Y"
						k1 = int(time.mktime(time.strptime(date_t1,pattern)))
						k2 = int(time.mktime(time.strptime(date_t2,pattern)))
						duration = (k1-k2)/(86400.0*365)
						temp_du[x[0]] = duration
						temp[x[0]] = val
					except Exception as e:
						pass
									
				for u,v in temp.items():
						if u  == 'larry page':
                                                                print u
						if u.endswith("s"):
							k = u[0:-1]
							if k in temp:
								v.extend(temp[k])
								del temp[k]
				 
				all_rates = defaultdict(list)
				count_f = lambda x:1 if len(x)>0 else 0
				for x,x1 in temp.items():
                                                rates = []
						val = temp[x]
                                           
                                                            
						#print x[0],val
						if len(val)>20:
							
							arival = sorted(val,reverse=False)
							path_len = 1000000
							temp_cats = []
							if x in c3:
								temp_cat = c3[x]
								#print x,temp_cat
								for y in temp_cat:
									if y in c1:
										temp_cats.append(c1[y])
							if len(temp_cats)>0:
								for x2 in temp_cats:					
									try:
										t = short_dict[x2]
										if t<path_len:
											path_len = t
									except Exception as e:								
											pass
								
							else:
								path_len = -1
							if path_len == 0:
								path_len = 1
							d = hawks_estimation(arival)
							#print d[0],d[1],d[2
							mus.append(d[0])
							alphas.append(d[1])
							betas.append(d[2])
                                                        for occur in range(len(arival)):
                                                                    x_1 = arival[:occur]
                                                                    rte = lambda_func(x_1,arival[occur],occur,d[0],d[1],d[2])
                                                                    rates.append(rte)
							try:
								if random.uniform(0,1)<0.9:
								    out = [1 for xx in range(len(arival))]
								    l_name = str(x).replace(" ","_")
		                                                    plt.figure(figsize=(20,10))
		                                                    m,s,b = plt.stem(arival,out,'-')
		                                                    plt.setp(b,'color','r','linewidth',1)
								    plt.xticks(arival,rotation='vertical')
								    plt.title(l_name+"Arrival")
		                                                    plt.savefig(l_name+"_.png")
								    plt.close()
								    plt.figure(figsize=(20,10))
		                                                    plt.plot(arival,rates,lw=2,marker="v",ms=6)
								    plt.xticks(arival,rotation='vertical')
								    plt.title(l_name+"_Rate")
		                                                    plt.savefig(l_name+".png")
								    plt.close()
							except:
								pass
							try:
								print x
	                                                        for xi,xj in zip(arival,rates):
									print xi,xj,
									print "\t"
								print "\n"
								print d[0],d[1],d[2]       
							except:
								pass
							#cn = [ y1 for y1 in temps[x] if y1<5]
							try:
								if path_len !=-1:
									cn = [ y1 for y1 in temps[str(x)] if y1<5]
									cn1 = [ y1 for y1 in temps[x] if y1>=5 and y1<10]
									cn2 = [ y1 for y1 in temps[x] if y1>=10 and y1<15]
									cn3 = [ y1 for y1 in temps[x] if y1>=15 and y1<20]
									cn4 = [ y1 for y1 in temps[x] if y1>=15 and y1<20]
									cn5 = [ y1 for y1 in temps[x] if y1>=20 and y1<25]
									cn6 = [ y1 for y1 in temps[x] if y1>=25 and y1<30]
									cn7 = [ y1 for y1 in temps[x] if y1>=30 and y1<35]
									cn8 = [ y1 for y1 in temps[x] if y1>=35 and y1<40]
									cn9 = [ y1 for y1 in temps[x] if y1>=40 and y1<50]
									#cn4 = [ y1 for y1 in temps[x] if y1==3]
									#cn5 = [ y1 for y1 in temps[x] if y1==4]
									#print x.replace(" ","_"),d[0],rtes[str(x)],path_len,len(val),len(set(temps[x])),count_f(cn)
									all_rates[x].extend((np.mean(rates),d[0],d[1],d[2],rtes[x],temp_du[x],path_len,len(val),len(set(temps[x])),count_f(cn),count_f(cn1),count_f(cn2),count_f(cn3),count_f(cn4),count_f(cn5),count_f(cn6),count_f(cn7),count_f(cn8),count_f(cn9)))
							except Exception as ex:
								print "Here",ex
				
				fa = open(sys.argv[2]+"_actual"+".csv","w")
				ft = open(sys.argv[2]+"_trivial"+".csv","w")

				my_temp=[]
                                print len(trivial)
                                for x,y in trivial.items():
                                    if x not in c3:
                                        del trivial[x]
                                    else:
                                        for j in iter(y):
                                            if j not in c3:
                                                y.remove(j)

                                print len(trivial)
				for x in trivial:
						if x not in my_temp and x in c3:
							#print x.replace(" ","_"),
                                                        ft.write(x.replace(" ","_"))
							fa.write(x.replace(" ","_"))
							for y in all_rates[x]:
								#print y,
								ft.write("\t"+str(y))
								fa.write("\t"+str(y))
							#print 
							ft.write("\n")
							fa.write("\n")
							for y1 in trivial[x]:
								if y1 not in my_temp:
									my_temp.append(y1)
								#print y1.replace(" ","_"),
								ft.write(y1.replace(" ","_"))
								for y in all_rates[y1]:
	                        					#print y,
									ft.write("\t"+str(y))
                                				#print
								ft.write("\n")
                                                        #print
							ft.write("\n")


                                print len(my_temp)
                                
				for x in all_rates:
                                        if len(all_rates[x])==0:
                                                continue
                                        else:
				            if x not in trivial:
						fa.write(x.replace(" ","_"))
						for y in all_rates[x]:
							fa.write("\t"+str(y))
						fa.write("\n")
				fa.close()
				ft.close()

				#pickle.dump(alphas,open("the_alphas.p","wb"))
				#pickle.dump(betas,open("the_betas.p","wb"))
				#pickle.dump(mus,open("the_mus.p","wb"))



'''
times.append(s)
while s<T:
	lmdb = lambda_func(times,s,mu,al,bi)
	u = rn.uniform(0,1)
	w = -np.log(u)/lmdb
	s=s+w
	atimes.append(s)
	rates.append(lmdb)
	#print s,lmdb
	D = rn.uniform(0,1)
	if D*lmdb <= lambda_func(times,s,mu,al,bi):
		times.append(s)


if times[-1]<=T:
		tm = times
else:
		tm = times[:-1]	

pickle.dump(tm,open("testing_syn_hawks.p","wb"))
params = hawks_estimation(tm)
print [float(x) for x in params]

#plt.plot(rates,lw=1)
#plt.plot(tm,np.ones(len(tm)),'ro',ms=4)
#plt.show()
