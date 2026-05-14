package mame

type Expr interface{ exprNode() }
type Stmt interface{ stmtNode() }

type NumLit struct{ Value int }
type FloatLit struct{ Value float64 }
type StrLit struct{ Value string }
type ArgRef struct{ Index Expr }
type NargExpr struct{}
type VarRef struct{ Name string }
type BinOp struct {
	Op   TokenType
	L, R Expr
}
type UnaryOp struct {
	Op TokenType
	X  Expr
}
type CallExpr struct {
	Name string
	Args []Expr
}

func (*NumLit) exprNode()   {}
func (*FloatLit) exprNode() {}
func (*StrLit) exprNode()   {}
func (*ArgRef) exprNode()   {}
func (*NargExpr) exprNode() {}
func (*VarRef) exprNode()   {}
func (*BinOp) exprNode()    {}
func (*UnaryOp) exprNode()  {}
func (*CallExpr) exprNode() {}

type AssignStmt struct {
	Name  string
	Value Expr
}
type ExprStmt struct{ X Expr }
type IfStmt struct {
	Cond Expr
	Then []Stmt
	Else []Stmt
}
type WhileStmt struct {
	Cond Expr
	Body []Stmt
}
type BreakStmt struct{}

func (*AssignStmt) stmtNode() {}
func (*ExprStmt) stmtNode()   {}
func (*IfStmt) stmtNode()     {}
func (*WhileStmt) stmtNode()  {}
func (*BreakStmt) stmtNode()  {}

type Program struct{ Stmts []Stmt }

func (c *Compiler) Parse() *Program {
	c.tokenPos = 0
	c.usesArg = false
	c.usesStrToInt = false
	c.usesStrToFloat = false
	c.usesFloatToStr = false
	c.usesPrint = false
	c.usesPrintStr = false
	prog := &Program{}
	for {
		for c.peek().Type == TOK_SEMI {
			c.consume(TOK_SEMI)
		}
		if c.peek().Type == TOK_EOF {
			break
		}
		prog.Stmts = append(prog.Stmts, c.parseStmt())
	}
	c.program = prog
	return prog
}

func (c *Compiler) parseStmt() Stmt {
	if c.peek().Type == TOK_IDENT && c.peek().Name == "if" {
		return c.parseIf()
	}
	if c.peek().Type == TOK_IDENT && c.peek().Name == "while" {
		return c.parseWhile()
	}
	if c.peek().Type == TOK_IDENT && c.peek().Name == "break" {
		c.consume(TOK_IDENT)
		return &BreakStmt{}
	}
	if c.peek().Type == TOK_IDENT && c.tokenPos+1 < len(c.tokens) && c.tokens[c.tokenPos+1].Type == TOK_ASSIGN {
		name := c.consume(TOK_IDENT).Name
		c.consume(TOK_ASSIGN)
		return &AssignStmt{Name: name, Value: c.parseExpr()}
	}
	return &ExprStmt{X: c.parseExpr()}
}

func (c *Compiler) parseIf() *IfStmt {
	c.consume(TOK_IDENT)
	cond := c.parseExpr()
	then := c.parseBlock()
	saved := c.tokenPos
	for c.peek().Type == TOK_SEMI {
		c.consume(TOK_SEMI)
	}
	var els []Stmt
	if c.peek().Type == TOK_IDENT && c.peek().Name == "else" {
		c.consume(TOK_IDENT)
		if c.peek().Type == TOK_IDENT && c.peek().Name == "if" {
			els = []Stmt{c.parseIf()}
		} else {
			els = c.parseBlock()
		}
	} else {
		c.tokenPos = saved
	}
	return &IfStmt{Cond: cond, Then: then, Else: els}
}

func (c *Compiler) parseWhile() *WhileStmt {
	c.consume(TOK_IDENT)
	cond := c.parseExpr()
	body := c.parseBlock()
	return &WhileStmt{Cond: cond, Body: body}
}

func (c *Compiler) parseBlock() []Stmt {
	c.consume(TOK_LBRACE)
	var stmts []Stmt
	for {
		for c.peek().Type == TOK_SEMI {
			c.consume(TOK_SEMI)
		}
		if c.peek().Type == TOK_RBRACE {
			break
		}
		stmts = append(stmts, c.parseStmt())
	}
	c.consume(TOK_RBRACE)
	return stmts
}

func (c *Compiler) parseExpr() Expr {
	left := c.parseSum()
	for {
		t := c.peek().Type
		if t != TOK_EQ && t != TOK_NE && t != TOK_LT && t != TOK_LE && t != TOK_GT && t != TOK_GE {
			break
		}
		op := c.consume(t).Type
		right := c.parseSum()
		left = &BinOp{Op: op, L: left, R: right}
	}
	return left
}

func (c *Compiler) parseSum() Expr {
	left := c.parseTerm()
	for c.peek().Type == TOK_PLUS || c.peek().Type == TOK_MINUS {
		op := c.consume(c.peek().Type).Type
		right := c.parseTerm()
		left = &BinOp{Op: op, L: left, R: right}
	}
	return left
}

func (c *Compiler) parseTerm() Expr {
	left := c.parseUnary()
	for c.peek().Type == TOK_MUL || c.peek().Type == TOK_DIV || c.peek().Type == TOK_MOD {
		op := c.consume(c.peek().Type).Type
		right := c.parseUnary()
		left = &BinOp{Op: op, L: left, R: right}
	}
	return left
}

func (c *Compiler) parseUnary() Expr {
	if c.peek().Type == TOK_PLUS {
		c.consume(TOK_PLUS)
		return c.parseUnary()
	}
	if c.peek().Type == TOK_MINUS {
		c.consume(TOK_MINUS)
		x := c.parseUnary()
		switch v := x.(type) {
		case *NumLit:
			return &NumLit{Value: -v.Value}
		case *FloatLit:
			return &FloatLit{Value: -v.Value}
		}
		return &UnaryOp{Op: TOK_MINUS, X: x}
	}
	return c.parseFactor()
}

func (c *Compiler) parseFactor() Expr {
	if c.peek().Type == TOK_NUM {
		return &NumLit{Value: c.consume(TOK_NUM).Value}
	}
	if c.peek().Type == TOK_FNUM {
		return &FloatLit{Value: c.consume(TOK_FNUM).Float}
	}
	if c.peek().Type == TOK_STRING {
		return &StrLit{Value: c.consume(TOK_STRING).Name}
	}
	if c.peek().Type == TOK_IDENT {
		name := c.consume(TOK_IDENT).Name
		if c.peek().Type == TOK_LPAREN {
			c.consume(TOK_LPAREN)
			var args []Expr
			if c.peek().Type != TOK_RPAREN {
				args = append(args, c.parseExpr())
			}
			c.consume(TOK_RPAREN)
			if name == "arg" {
				if len(args) != 1 {
					panic("arg takes 1 argument")
				}
				c.usesArg = true
				return &ArgRef{Index: args[0]}
			}
			if name == "narg" {
				if len(args) != 0 {
					panic("narg takes no arguments")
				}
				c.usesArg = true
				return &NargExpr{}
			}
			if name == "int" {
				if len(args) != 1 {
					panic("int takes 1 argument")
				}
				c.usesStrToInt = true
			}
			if name == "float" {
				if len(args) != 1 {
					panic("float takes 1 argument")
				}
				c.usesStrToFloat = true
			}
			if name == "str" {
				if len(args) != 1 {
					panic("str takes 1 argument")
				}
				c.usesIntToStr = true
				c.usesFloatToStr = true
			}
			if name == "print" || name == "println" {
				if len(args) == 1 {
					if _, ok := args[0].(*StrLit); ok {
						c.usesPrintStr = true
					} else {
						c.usesPrint = true
					}
				}
				if name == "println" {
					c.usesPrintStr = true
				}
			}
			return &CallExpr{Name: name, Args: args}
		}
		return &VarRef{Name: name}
	}
	if c.peek().Type == TOK_LPAREN {
		c.consume(TOK_LPAREN)
		e := c.parseExpr()
		c.consume(TOK_RPAREN)
		return e
	}
	panic("unexpected token")
}
