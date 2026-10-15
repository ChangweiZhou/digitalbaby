"""Independent environment-only truth and group checker; never imported by learner."""
import ast,random

def independent_answer(prompt,seed):
    pi=list(range(256));random.Random(seed+918273).shuffle(pi);inv={v:k for k,v in enumerate(pi)}
    text=bytes(inv[b] for b in bytes.fromhex(prompt)).decode()
    assert text.endswith('=')
    def evaluate(t):
        if isinstance(t,ast.Constant) and type(t.value) is int:return t.value
        if isinstance(t,ast.UnaryOp) and isinstance(t.op,ast.USub):return -evaluate(t.operand)
        assert isinstance(t,ast.BinOp)
        a,b=evaluate(t.left),evaluate(t.right)
        if isinstance(t.op,ast.Add):return a+b
        if isinstance(t.op,ast.Sub):return a-b
        if isinstance(t.op,ast.Mult):return a*b
        assert isinstance(t.op,ast.Div) and b!=0 and a%b==0
        return a//b
    value=evaluate(ast.parse(text[:-1],mode='eval').body)
    return bytes(pi[b] for b in str(value).encode()).hex()

def independent_group(prompt,seed):
    pi=list(range(256));random.Random(seed+918273).shuffle(pi);inv={v:k for k,v in enumerate(pi)}
    text=bytes(inv[b] for b in bytes.fromhex(prompt)).decode()[:-1]
    names={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}
    def value(t):
        if isinstance(t,ast.Constant):return t.value
        if isinstance(t,ast.UnaryOp):return -value(t.operand)
        a,b=value(t.left),value(t.right);op=names[type(t.op)]
        return a+b if op=='+' else a-b if op=='-' else a*b if op=='*' else a//b
    def canon(t):
        if not isinstance(t,ast.BinOp):return ('leaf',value(t))
        op=names[type(t.op)];kids=[]
        def add(x):
            if op in ('+','*') and isinstance(x,ast.BinOp) and names[type(x.op)]==op:add(x.left);add(x.right)
            else:kids.append(canon(x))
        add(t.left);add(t.right)
        if op in ('+','*'):kids.sort(key=repr)
        return (op,*kids)
    t=ast.parse(text,mode='eval').body
    if not isinstance(t.left,ast.BinOp) and not isinstance(t.right,ast.BinOp):
        a,b,y=value(t.left),value(t.right),value(t);op=names[type(t.op)]
        if op in ('+','-'):
            s=sorted((abs(a),abs(b),abs(y)));return repr(('SUM',s[0],s[1]))
        pair=sorted((abs(a),abs(b)) if op=='*' else (abs(b),abs(y)));return repr(('PRODUCT',*pair))
    return repr(canon(t))
