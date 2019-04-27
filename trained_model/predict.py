# -*- coding: utf-8 -*-
import datetime
from pre_processing import *
flag = ""
print("\n输入你自己的句子，或者直接回车查看默认示例：")
sentence = input()

while flag != 'exit':
	if not sentence:
		sentence = u"7月19日至28日，国家主席习近平将对阿联酋、塞内加尔、卢旺达和南非进行国事访问，出席在南非约翰内斯堡举行的金砖国家领导人第十次会晤，"\
					"过境毛里求斯并进行友好访问。此次中东非洲之行是习近平连任国家主席后的首次出访，是国际形势深刻演变背景下中国面向发展中国家的重"\
					"大外交行动，意义重大，世界瞩目。"
		print("示例句子为：" + sentence)
	start_time = datetime.datetime.now().microsecond				
	entity_name, entity_tags = predict(sentence)
	print("**********识别结果**********")
	print_result(entity_name, entity_tags)
	end_time = datetime.datetime.now().microsecond
	run_time = (end_time - start_time)/1000
	print("**********finished in %.2f ms**********" % run_time)

	print("\n继续或输入'exit'退出")
	flag = input()
	if flag != 'exit':
		sentence = flag
	else:
		flag = flag

print('\n再见')
	