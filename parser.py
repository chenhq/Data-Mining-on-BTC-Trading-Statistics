from bs4 import BeautifulSoup
import html5lib
import requests
import sys
import re
import json

filename = u'parsed.html'
#print(filename)
soup = BeautifulSoup(open(filename,'rt',encoding='utf-8'), 'html5lib')
temp = soup.decode('utf-8')
name = soup.select('h4')
#alpha 1-82
pattern1 = re.compile('<pre.*?<ul>.*?</ul>.*?<ul>.*?</ul>.*?<ul>(.*?)</ul>', re.S)
#alpha 83-101
#pattern2 = re.compile('<pre.*?<ul>.*?</ul>.*?<ul>.*?</ul>.*?<ul>(.*?)</ul>', re.S)
expression1 = re.findall(pattern1, temp)
#expression2 = re.findall(pattern2, temp)


'''
idx = 1
for item in expression1:
	print(idx)
	idx += 1
	print (re.sub(' ', '', re.sub('<li>', '', re.sub('</li>', '', re.sub('\n', '', item)))))
'''
#expression = expression1.append(expression2)
#print(items[0])

#expression = soup.select('br > ul > li')
    
alphas = []
    
# print(name)


for name, expression in zip(name, expression1):
    alpha = {
        'name': name.get_text(),
        'expression': re.sub(' ', '', re.sub('<li>', '', re.sub('</li>', '', re.sub('\n', '', expression))))
    }
    #print (alpha)
    alphas.append(alpha)

formulas = []
for alpha in alphas:
    formulas.append(alpha['name']+'='+alpha['expression'])

with open('alpha101.txt','w') as f:
	for formula in formulas:
		#print(formula)
		formula = re.sub('correlation', 'corr', formula)
		formula = re.sub('covariance','COVIANCE', formula)
		formula = re.sub('decay_linear', 'DECAYLINEAR', formula)
		formula = re.sub('ts_min','TSMIN', formula)
		formula = re.sub('ts_max','TSMAX', formula)
		formula = re.sub('Ts_Rank','TSRANK', formula)
		formula = re.sub('stddev', 'STD', formula)
		formula = re.sub('returns','RET', formula)
		formula = re.sub('&LT;', '<', formula)
		formula = re.sub('\(\u672a\u5b9e\u73b0\)', '', formula)
		formula = formula.upper()
		print(formula)
		f.write(json.dumps(formula)+'\n')
	#f.write(json.dumps(formula))


