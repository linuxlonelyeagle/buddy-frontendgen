BUDDY_FRONT_END_GEN := ../../build/bin/buddy-frontendgen   

frontendgen-emit-ast:
	@${BUDDY_FRONT_END_GEN} -f ./example.fegen -emit=ast 

frontendgen-emit-antlr:
	@${BUDDY_FRONT_END_GEN} -f ./example.fegen -emit=antlr -o Toy.g4


frontendgen-emit-td:
	@${BUDDY_FRONT_END_GEN} -f ./example.fegen -emit=td 

frontendgen-emit-all:
	@${BUDDY_FRONT_END_GEN} -f ./example.fegen -emit=all -o Toy.g4