/*
  expr        := factor [ ('+' | '-') factor ]*
  factor      := ['-']? term [ ('*' | '/' | '') ['-']? term ]*                    // the second ['-']? only if the operator is not ''
  term        := atom [ '^' ['-']? atom ]*
  atom        := identifier
               | '(' expr ')' ['\'']*
               | identifier ['\'']* '(' expr ')'
               | number

  identifier  := [a-zA-Z_] [a-zA-Z0-9_]*
  number      := [0-9]* '.' [0-9]*
               | [0-9]+
*/

class Ptr {
  constructor(i = 0){
    this.i = i;
  }
}
class Expr {
  static ADD =0;
  static SUB =1;
  static MUL =2;
  static DIV =3;
  static SUB_UN =4;
  static POW =5;
  static FUNC =6;
  static VAR =7;
  static NUM =8;
  static DIFF =9;

  static build_in_functions = ["sin", "cos", "tan", "arcsin", "arccos", "arctan", "ln", "log", "sqrt"];

  constructor(type = -1, children = [], content = "", diffs = 0){
    this.type = type;
    this.content = content;
    this.children = children;
    this.diffs = diffs; // only applies if type == FUNC

    this.last_eval = 0;
    this.eval = 0;
  }

  copy(){
    var expr = new Expr(this.type);
    expr.content = this.content;
    expr.diffs = this.diffs;
    expr.last_eval = this.last_eval;
    expr.eval = this.eval;
    for (const child of this.children) expr.children.push(child.copy());
    return expr;
  }

  copyFrom(other){
    this.type = other.type;
    this.content = other.content;
    this.diffs = other.diffs;
    this.isConstant = other.isConstant;
    this.last_eval = other.last_eval;
    this.eval = other.eval;
    this.children = [];
    for (const child of other.children) this.children.push(child.copy());
  }

  // only use when there are no parameters or references to other functions (besides the built-in ones)
  evaluate(x, checkBreak = true){
    var evals = [];
    for (var i = 0; i < this.children.length; ++i){
      if (!this.children[i].evaluate(x, checkBreak)) return false;
      evals[i] = this.children[i].eval;
    }
    this.last_eval = this.eval;
    switch (this.type){
      case Expr.ADD:
        this.eval = evals[0] + evals[1];
        break;
      case Expr.SUB:
        this.eval = evals[0] - evals[1];
        break;
      case Expr.MUL:
        this.eval = evals[0] * evals[1];
        break;
      case Expr.DIV:
        this.eval = evals[0] / evals[1];
        if (checkBreak && Math.sign(this.children[1].last_eval) != Math.sign(evals[1]))
          return false;
        break;
      case Expr.POW:
        this.eval = Math.pow(evals[0], evals[1]);
        break;
      case Expr.SUB_UN:
        this.eval = -evals[0];
        break;
      case Expr.FUNC:
        switch (this.content){
          case "sin": this.eval = Math.sin(evals[0]); break;
          case "cos": this.eval = Math.cos(evals[0]); break;
          case "tan":
            this.eval = Math.tan(evals[0]);
            if (checkBreak && Math.floor(this.children[0].last_eval / Math.PI - 0.5) != Math.floor(evals[0] / Math.PI - 0.5))
              return false;
            break;
          case "arcsin": this.eval = Math.asin(evals[0]); break;
          case "arccos": this.eval = Math.acos(evals[0]); break;
          case "arctan": this.eval = Math.atan(evals[0]); break;
          case "log": case "ln":
            this.eval = Math.log(evals[0]);
            break;
          case "sqrt": this.eval = Math.sqrt(evals[0]); break;
          default: this.eval = NaN;
        }
        break;
      case Expr.VAR:
        if (this.content == "x") this.eval = x;
        else return NaN;
        break;
      case Expr.NUM:
        this.eval = this.content;
        break;
    }
    return !isNaN(this.eval);
  }

  // only use when there are no parameters or references to other functions (besides the built-in ones)
  evaluateConstants(){
    for (const child of this.children) child.evaluateConstants();
    if (this.type == Expr.SUB_UN && this.children[0].type == Expr.NUM){
      this.type = Expr.NUM;
      this.content = -this.children[0].content;
      this.children = [];
      return;
    }
    if (this.type < 6 && this.children[0].type == Expr.NUM && this.children[1].type == Expr.NUM){
      switch (this.type){
        case Expr.ADD: this.content = this.children[0].content + this.children[1].content; break;
        case Expr.SUB: this.content = this.children[0].content - this.children[1].content; break;
        case Expr.MUL: this.content = this.children[0].content * this.children[1].content; break;
        case Expr.DIV: this.content = this.children[0].content / this.children[1].content; break;
        case Expr.POW: this.content = Math.pow(this.children[0].content, this.children[1].content); break;
      }
      this.type = Expr.NUM;
      this.children = [];
      return;
    }
    if (this.type == 6 && this.children[0].type == Expr.NUM){
      this.type = Expr.NUM;
      switch (this.content){
        case "sin": this.content = Math.sin(this.children[0].content); break;
        case "cos": this.content = Math.cos(this.children[0].content); break;
        case "tan": this.content = Math.tan(this.children[0].content); break;
        case "arcsin": this.content = Math.asin(this.children[0].content); break;
        case "arccos": this.content = Math.acos(this.children[0].content); break;
        case "arctan": this.content = Math.atan(this.children[0].content); break;
        case "log": case "ln":
          this.content = Math.log(this.children[0].content);
          break;
        case "sqrt": this.content = Math.sqrt(this.children[0].content); break;
        default: this.content = NaN;
      }
      this.children = [];
    }
  }

  substituteX(sub_expr){
    if (this.type == Expr.VAR && this.content == "x"){
      this.copyFrom(sub_expr);
    } else for (const child of this.children) child.substituteX(sub_expr);
  }

  substituteFunction(function_name, function_expr){
    for (const child of this.children) child.substituteFunction(function_name, function_expr);
    if (this.type == Expr.FUNC && this.content == function_name){
      for (var i = 0; i < this.diffs; ++i) function_expr = function_expr.differentiated.simplified;
      var evaluate = this.children[0].copy();
      this.copyFrom(function_expr);
      this.substituteX(evaluate);
    }
  }

  substituteParameters(params){
    for (const child of this.children) child.substituteParameters(params);
    if (this.type == Expr.VAR && params[this.content] != undefined){
      this.type = Expr.NUM;
      this.content = params[this.content];
    }
  }

  // make sure there are no references to functions
  resolveDIFF(){
    for (const child of this.children) child.resolveDIFF();
    if (this.type == Expr.FUNC && this.diffs){
      var expr = new Expr(Expr.FUNC, [new Expr(Expr.VAR, [], "x")], this.content);
      for (var i = 0; i < this.diffs; ++i) expr = expr.differentiated.simplified;
      expr.substituteX(this.children[0]);
      this.copyFrom(expr);
      return;
    }
    if (this.type == Expr.DIFF){
      for (var i = 0; i < this.content; ++i) this.children[0] = this.children[0].differentiated.simplified;
      this.type = this.children[0].type;
      this.content = this.children[0].content;
      this.diffs = this.children[0].diffs;
      this.children = this.children[0].children;
    }
  }

  getDependencies(variables, functions){
    if (this.type == Expr.VAR && !variables.includes(this.content)) variables.push(this.content);
    if (this.type == Expr.FUNC && !Expr.build_in_functions.includes(this.content) && !functions.includes(this.content)) functions.push(this.content);
    for (const child of this.children) child.getDependencies(variables, functions);
  }

  get differentiated(){
    switch (this.type){
      case Expr.ADD:
        return new Expr(Expr.ADD, [this.children[0].differentiated, this.children[1].differentiated]);
      case Expr.SUB:
        return new Expr(Expr.SUB, [this.children[0].differentiated, this.children[1].differentiated]);
      case Expr.MUL:
        return new Expr(Expr.ADD, [
          new Expr(Expr.MUL, [this.children[0].copy(), this.children[1].differentiated]),
          new Expr(Expr.MUL, [this.children[0].differentiated, this.children[1].copy()])
        ]);
      case Expr.DIV:
        return new Expr(Expr.DIV, [
          new Expr(Expr.SUB, [
            new Expr(Expr.MUL, [this.children[1].copy(), this.children[0].differentiated]),
            new Expr(Expr.MUL, [this.children[0].copy(), this.children[1].differentiated])
          ]),
          new Expr(Expr.POW, [this.children[1].copy(), Expr.number(2)])
        ]);
      case Expr.POW:
        if (this.children[1].type == Expr.NUM){
          return new Expr(Expr.MUL, [
            Expr.number(this.children[1].content),
            new Expr(Expr.POW, [
              this.children[0].copy(),
              Expr.number(this.children[1].content - 1)
            ])
          ]);
        }
        return new Expr(Expr.MUL, [
          new Expr(Expr.ADD, [
            new Expr(Expr.MUL, [
              this.children[1].differentiated,
              new Expr(Expr.FUNC, [this.children[0].copy()], "ln")
            ]),
            new Expr(Expr.DIV, [
              new Expr(Expr.MUL, [this.children[1].copy(), this.children[0].differentiated]),
              this.children[0].copy()
            ])
          ]),
          new Expr(Expr.POW, [this.children[0].copy(), this.children[1].copy()])
        ]);
      case Expr.SUB_UN:
        return new Expr(Expr.SUB_UN, [this.children[0].differentiated]);
      case Expr.FUNC:
        switch (this.content){
          case "sin":
            return new Expr(Expr.MUL, [
              new Expr(Expr.FUNC, [this.children[0].copy()], "cos"),
              this.children[0].differentiated
            ]);
          case "cos":
            return new Expr(Expr.MUL, [
              new Expr(Expr.SUB_UN, [new Expr(Expr.FUNC, [this.children[0].copy()], "sin")]),
              this.children[0].differentiated
            ]);
          case "tan":
            return new Expr(Expr.DIV, [
              this.children[0].differentiated,
              new Expr(Expr.POW, [
                new Expr(Expr.FUNC, [this.children[0].copy()], "cos"),
                Expr.number(2)
              ])
            ]);
          case "arcsin":
            return new Expr(Expr.DIV, [
              this.children[0].differentiated,
              new Expr(Expr.FUNC, [
                new Expr(Expr.SUB, [
                  Expr.number(1),
                  new Expr(Expr.POW, [
                    this.children[0].copy(),
                    Expr.number(2)
                  ])
                ])
              ], "sqrt")
            ]);
          case "arccos":
            return new Expr(Expr.SUB_UN, [new Expr(Expr.DIV, [
              this.children[0].differentiated,
              new Expr(Expr.FUNC, [
                new Expr(Expr.SUB, [
                  Expr.number(1),
                  new Expr(Expr.POW, [
                    this.children[0].copy(),
                    Expr.number(2)
                  ])
                ])
              ], "sqrt")
            ])]);
          case "arctan":
            return new Expr(Expr.DIV, [
              this.children[0].differentiated,
              new Expr(Expr.ADD, [
                Expr.number(1),
                new Expr(Expr.POW, [this.children[0].copy(), Expr.number(1)])
              ])
            ]);
          case "log": case "ln":
            return new Expr(Expr.DIV, [
              this.children[0].differentiated,
              this.children[0].copy()
            ]);
          default:
            return new Expr(Expr.MUL, [
              new Expr(Expr.FUNC, [this.children[0].copy()], this.content, this.diffs + 1),
              this.children[0].differentiated
            ]);
        }
        break;
      case Expr.VAR:
        return Expr.number(1 * (this.content == "x"));
      case Expr.NUM:
        return Expr.number(0);
    }
  }

  get simplified(){
    for (var i = 0; i < this.children.length; ++i) this.children[i] = this.children[i].simplified;
    switch (this.type){
      case Expr.ADD:
        if (this.children[0].isNumber(0)) return this.children[1].copy();
        if (this.children[1].isNumber(0)) return this.children[0].copy();
        break;
      case Expr.SUB:
        if (this.children[0].isNumber(0)) return new Expr(Expr.SUB_UN, [this.children[1].copy()]);
        if (this.children[1].isNumber(0)) return this.children[0].copy();
        break;
      case Expr.MUL:
        if (this.children[0].isNumber(0) || this.children[1].isNumber(0)) return Expr.number(0);
        if (this.children[0].isNumber(1)) return this.children[1].copy();
        if (this.children[1].isNumber(1)) return this.children[0].copy();
        break;
      case Expr.DIV:
        if (this.children[0].isNumber(0)) return Expr.number(0);
        if (this.children[1].isNumber(1)) return this.children[0].copy();
        break;
      case Expr.POW:
        if (this.children[1].isNumber(0)) return Expr.number(1);
        if (this.children[0].isNumber(0)) return Expr.number(0);
        if (this.children[1].isNumber(1)) return this.children[0].copy();
        if (this.children[0].isNumber(1)) return Expr.number(1);
        break;
      case Expr.SUB_UN:
        if (this.children[0].isNumber(0)) return Expr.number(0);
        if (this.children[0].type == Expr.NUM) return Expr.number(-this.children[0].content);
        if (this.children[0].type == Expr.SUB_UN) return this.children[0].children[0].copy();
        break;
      case Expr.FUNC: break;
      case Expr.VAR: break;
      case Expr.NUM: break;
      case Expr.DIFF: break;
    }
    return this.copy();
  }

  toString(){
    var strs = [];
    for (const child of this.children){
      var str = child.toString();
      if (this.type < 6 && child.type < this.type) str = "(" + str + ")";
      strs.push(str);
    }
    switch (this.type){
      case Expr.ADD:
        return strs[0] + " + " + strs[1];
      case Expr.SUB:
        return strs[0] + " - " + strs[1];
      case Expr.MUL:
        return strs[0] + " * " + strs[1];
      case Expr.DIV:
        return strs[0] + " / " + strs[1];
      case Expr.POW:
        return strs[0] + " ^ " + strs[1];
      case Expr.SUB_UN:
        return "-" + strs[0];
      case Expr.FUNC:
        return this.content + ("'".repeat(this.diffs)) + "(" + strs[0] + ")";
      case Expr.VAR:
        return this.content;
      case Expr.NUM:
        return this.content + "";
      case Expr.DIFF:
        return "(" + strs[0] + ")" + "'".repeat(this.content);
    }
  }

  static read(str){
    var expr = new Expr();
    var idx = new Ptr(0);
    if (acceptExpr(str, idx, expr)){
      skipSpaces(str, idx);
      if (idx.i == str.length){
        return expr;
      }
    }
    return null;
  }

  static number(n){
    var expr = new Expr(Expr.NUM);
    expr.content = n;
    return expr;
  }

  isNumber(n){
    return this.type == Expr.NUM && this.content == n;
  }
}
class ExprSystem {
  constructor(){
    this.raw_exprs = [];
    this.exprs = [];
    this.param_list = [];
    this.param_dict = {};
    this.function_dict = {};
    this.colors = [];
  }

  drawGraphs(win, draw_bools = new Array(colors.length).fill(true)){
    var dx = (win.window.Xmax - win.window.Xmin) / canvas1.width * 2;
    for (var i = 0; i < this.exprs.length; ++i){
      if (!draw_bools[i]) continue;
      this.exprs[i].evaluate(win.window.Xmin, false);
      var last = this.exprs[i].eval, current;
      for (var x = win.window.Xmin + dx; x <= win.window.Xmax; x += dx){
        var b = this.exprs[i].evaluate(x);
        current = this.exprs[i].eval;
        if (b && !isNaN(last) && !isNaN(current)) win.drawLine(new Vector2(x - dx, last), new Vector2(x, current), this.colors[i]);
        last = current;
      }
    }
  }

  // ["x + 8 / x"], ["y0"], {'a': 1.23}
  static compileExprSystem(strs, function_names, colors, params = {}, error_message = null){
    if (strs.length != function_names.length){
      if (error_message != null) error_message.i = "number of expressions and number of function names are not equal";
      return null;
    }
    var system = new ExprSystem();
    for (var i = 0; i < colors.length; ++i) system.colors[i] = colors[i];
    for (const key of Object.keys(system.param_dict = params)) system.param_list.push(key);
    var dependency_graph = new DependencyGraph(function_names);
    for (var i = 0; i < strs.length; ++i){
      var expr = Expr.read(strs[i]);
      if (expr == null){
        strs.splice(i, 1);
        function_names.splice(i, 1);
        system.colors.splice(i, 1);
        --i;
        continue;
      }
      system.raw_exprs[i] = expr;
      var variables = [], functions = [];
      system.raw_exprs[i].getDependencies(variables, functions);
      for (const v of variables){
        if (v != "x" && !system.param_list.includes(v)){
          if (error_message != null) error_message.i = "in '" + function_names[i] + "': variable '" + v + "' is not defined";
          return null;
        }
      }
      for (const f of functions){
        if (!function_names.includes(f)){
          if (error_message != null) error_message.i = "in '" + function_names[i] + "': function '" + f + "' is not defined";
          return null;
        }
      }
      dependency_graph.pushEdges(function_names[i], functions);
    }
    if (!dependency_graph.check(error_message)){
      if (error_message != null) error_message.i = "dependency error: " + error_message.i;
      return null;
    }
    for (var i = 0; i < function_names.length; ++i){
      for (var j = 0; j < system.raw_exprs.length; ++j){
        if (i != j) system.raw_exprs[j].substituteFunction(function_names[i], system.raw_exprs[i]);
      }
      system.raw_exprs[i].resolveDIFF();
    }
    for (var i = 0; i < system.raw_exprs.length; ++i){
      system.exprs[i] = system.raw_exprs[i].copy();
      system.exprs[i].substituteParameters(params);
      system.exprs[i].evaluateConstants();
      system.exprs[i] = system.exprs[i].simplified;
    }
    return system;
  }
}
class DependencyGraph {
  constructor(node_names){
    this.node_to_idx = {};
    this.matrix = [];
    for (var i = 0; i < node_names.length; ++i){
      this.matrix[i] = [];
      this.node_to_idx[node_names[i]] = i;
      for (var j = 0; j < node_names.length; ++j) this.matrix[i][j] = false;
    }
    this.node_names = node_names;
  }

  pushEdges(node_name, child_node_names){
    for (var name of child_node_names)
      this.matrix[this.node_to_idx[node_name]][this.node_to_idx[name]] = true;
  }

  check(message = null){
    for (var i = 0; i < this.matrix.length; ++i){
      var visited = new Array(this.matrix.length).fill(false);
      visited[i] = true;
      if (!this.checkPrivate(message, i, visited)) return false;
    }
    return true;
  }

  checkPrivate(message, current, visited){
    for (var i = 0; i < this.matrix.length; ++i){
      if (!this.matrix[current][i]) continue;
      if (!visited[i]){
        visited[i] = true;
        if (!this.checkPrivate(message, i, visited)){
          if (message != null) message.i = this.node_names[current] + " <- " + message.i;
          return false;
        }
        visited[i] = false;
      } else {
        if (message != null) message.i = this.node_names[current] + " <- " + this.node_names[i];
        return false;
      }
    }
    return true;
  }
}

function isAlph(c){
  if (c == undefined) return false;
  var ci = c.charCodeAt();
  return (ci >= 97 && ci <= 122) || (ci >= 65 && ci <= 90);
}
function isDigit(c){
  if (c == undefined) return false;
  var ci = c.charCodeAt();
  return ci >= 48 && ci <= 57;
}
function skipSpaces(str, idx){
  while (str[idx.i] == ' ' || str[idx.i] == '\t' || str[idx.i] == '\n') ++idx.i;
}
function accept(str, idx, target, ptr = null){
  var temp = idx.i;
  skipSpaces(str, idx);
  if (str.substr(idx.i, target.length) == target){
    idx.i += target.length;
    if (ptr != null) ptr.i = target;
    return true;
  }
  idx.i = temp;
  return false;
}

function acceptExpr(str, idx, node){
  var temp = idx.i;
  if (acceptFactor(str, idx, node)){
    var ptr = new Ptr();
    var next = new Expr();
    var temp1 = idx.i;
    while ((accept(str, idx, "+", ptr) || accept(str, idx, "-", ptr)) && acceptFactor(str, idx, next)){
      node.copyFrom(new Expr((ptr.i == "+") ? Expr.ADD : Expr.SUB, [node.copy(), next]));
      next = new Expr();
      temp1 = idx.i;
    }
    idx.i = temp1;
    return true;
  }
  idx.i = temp;
  return false;
}
function acceptFactor(str, idx, node){
  var temp = idx.i;
  var isMinus = accept(str, idx, "-");
  if (acceptTerm(str, idx, node)){
    var ptr = new Ptr();
    var next = new Expr();
    var temp1 = idx.i;
    var anotherMinus = false;
    while ((accept(str, idx, "*", ptr) || accept(str, idx, "/", ptr) || (ptr.i = "") == "") && ((ptr.i.length && (anotherMinus = accept(str, idx, "-"))) || true) && acceptTerm(str, idx, next)){
      var new_node = new Expr((ptr.i == "/") ? Expr.DIV : Expr.MUL, [node.copy(), next]);
      node.copyFrom(anotherMinus ? new Expr(Expr.SUB_UN, [new_node]) : new_node);
      anotherMinus = false;
      next = new Expr();
      temp1 = idx.i;
    }
    idx.i = temp1;
    if (isMinus){
      var t = node.copy();
      node.content = "";
      node.type = Expr.SUB_UN;
      node.children = [t];
    }
    return true;
  }
  idx.i = temp;
  return false;
}
function foldAtoms(atoms, idx =0){
  if (idx == atoms.length - 1) return atoms[idx];
  return new Expr(Expr.POW, [atoms[idx], foldAtoms(atoms, idx + 1)]);
}
function acceptTerm(str, idx, node){
  var temp = idx.i;
  var next = new Expr();
  if (acceptAtom(str, idx, next)){
    var atoms = [next];
    next = new Expr();
    var temp1 = idx.i;
    var b;
    while (accept(str, idx, "^") && ((b = accept(str, idx, "-")) || true) && acceptAtom(str, idx, next)){
      atoms.push(b ? new Expr(Expr.SUB_UN, [next]) : next);
      next = new Expr();
      temp1 = idx.i;
    }
    idx.i = temp1;
    node.copyFrom(foldAtoms(atoms));
    return true;
  }
  idx.i = temp;
  return false;
}
function acceptAtom(str, idx, node){
  var temp = idx.i;
  var iden = new Ptr("");
  var isIden = acceptIdentifier(str, idx, iden);
  var numDiff = 0;
  if (isIden){
    while (accept(str, idx, "'")) ++numDiff;
  }
  if (accept(str, idx, "(")){
    if (acceptExpr(str, idx, node) && accept(str, idx, ")")){
      if (!isIden){
        var n = 0;
        while (accept(str, idx, "'")) ++n;
        if (n) node.copyFrom(new Expr(Expr.DIFF, [node.copy()], n));
      } else {
        node.copyFrom(new Expr(Expr.FUNC, [node.copy()], iden.i, numDiff));
      }
      return true;
    }
    idx.i = temp;
    return false;
  }
  if (isIden){
    if (numDiff){
      idx.i = temp;
      return false;
    }
    node.content = iden.i;
    node.type = Expr.VAR;
    node.children = [];
    return true;
  }
  if (acceptNumber(str, idx, iden)){
    node.content = iden.i;
    node.type = Expr.NUM;
    node.children = [];
    return true;
  }
  idx.i = temp;
  return false;
}

function acceptIdentifier(str, idx, ptr){
  var temp = idx.i;
  skipSpaces(str, idx);
  var start = idx.i;
  if (isAlph(str[idx.i]) || str[idx.i] == '_'){
    ++idx.i;
    while (isAlph(str[idx.i]) || str[idx.i] == '_' || isDigit(str[idx.i])) ++idx.i;
    ptr.i = str.substr(start, idx.i - start);
    return true;
  }
  idx.i = temp;
  return false;
}
function acceptNumber(str, idx, ptr){
  var temp = idx.i;
  skipSpaces(str, idx);
  var num = 0;
  var b = false;
  while (isDigit(str[idx.i])){
    b = true;
    num = num * 10 + (str[idx.i] * 1);
    ++idx.i;
  }
  if (str[idx.i] == '.'){
    b = true;
    ++idx.i;
    var power = 0.1;
    while (isDigit(str[idx.i])){
      num += str[idx.i] * power;
      power /= 10;
      ++idx.i;
    }
  }
  if (b){
    ptr.i = num;
    return true;
  }
  idx.i = temp;
  return false;
}
