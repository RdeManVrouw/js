class Vector2 {
  constructor(x, y){
    this.x = x;
    this.y = y;
  }
  Multpl(x){
    return new Vector2(this.x * x, this.y * x);
  }
  Add(v){
    return new Vector2(this.x + v.x, this.y + v.y);
  }
  Sub(v){
    return new Vector2(this.x - v.x, this.y - v.y);
  }
  Dist(v){
    return Math.hypot(this.x - v.x, this.y - v.y);
  }
  Dist2(v){
    return Math.sq(this.x - v.x) + Math.sq(this.y - v.y);
  }
  Dot(v){
    return this.x * v.x + this.y * v.y;
  }
  Crs(v){
    return this.x * v.y - this.y * v.x;
  }
  rotate(phi){
    return new Vector2(this.x * Math.cos(phi) - this.y * Math.sin(phi), this.x * Math.sin(phi) + this.y * Math.cos(phi));
  }
  static Angle(phi){
    return new Vector2(Math.cos(phi), Math.sin(phi));
  }
  get magnitude(){
    return Math.hypot(this.x, this.y);
  }
  get magnitude2(){
    return this.x * this.x + this.y * this.y;
  }
  get normalized(){
    return this.Multpl(1/this.magnitude);
  }
  static get zero(){
    return new Vector2(0, 0);
  }
}

class Polygon {
  constructor(verteces, color = "brown"){
    this.verteces = verteces;
    this.color = color;
  }

  Multpl(x){
    for (var i = 0; i < this.verteces.length; i++){
      this.verteces[i].x *= x;
      this.verteces[i].y *= x;
    }
  }

  intersectsCircle(c, r){
    for (var i = 0; i < this.verteces.length; i++){
      if (c.Dist2(this.verteces[i]) < r * r) return true;
      var p12 = this.verteces[(i + 1) % this.verteces.length].Sub(this.verteces[i]);
      var t = -p12.Dot(this.verteces[i].Sub(c)) / p12.magnitude2;
      if (t >= 0 && t <= 1 && this.verteces[i].Sub(c).Add(p12.Multpl(t)).magnitude2 < r * r) return true;
    }
    return false;
  }

  rotate(phi){
    for (var i = 0; i < this.verteces.length; i++) this.verteces[i] = this.verteces[i].rotate(phi);
  }

  static random(n){
    var ver = [];
    for (var i = 0; i < n; i++){
      ver[i] = Vector2.Angle((i + 0.8 * (Math.random() - 0.5)) * 2 * Math.PI / n, 1 + 0.5 * (Math.random() - 0.5));
    }
    return new Polygon(ver);
  }
}
class Vector3 {
  constructor(x, y, z){
    this.x = x;
    this.y = y;
    this.z = z;
  }
  m(m){
    return m.col1.Multpl(this.x).Add(m.col2.Multpl(this.y)).Add(m.col3.Multpl(this.z));
  }
  Add(v){
    return new Vector3(this.x + v.x, this.y + v.y, this.z + v.z);
  }
  Sub(v){
    return new Vector3(this.x - v.x, this.y - v.y, this.z - v.z);
  }
  Multpl(x){
    return new Vector3(this.x * x, this.y * x, this.z * x);
  }
  Dot(v){
    return this.x*v.x+this.y*v.y+this.z*v.z;
  }
  Crs(v){
    return new Vector3(this.y*v.z-this.z*v.y, this.z*v.x-this.x*v.z, this.x*v.y-this.y*v.x);
  }
  Bounce(n){
    return this.Add(n.Multpl(-2*n.Dot(this)));
  }
  Rot(q){
    return q.Prod(this.quat).Prod(q.conj).v;
  }
  Dist(v){
    return Math.hypot(this.x - v.x, this.y - v.y, this.z - v.z);
  }
  get magnitude(){
    return Math.hypot(this.x, this.y, this.z);
  }
  get magnitude2(){
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }
  get normalized(){
    return this.Multpl(1/this.magnitude);
  }
  get quat(){
    return new Quaternion(0, this);
  }
  static get zero(){
    return new Vector3(0, 0, 0);
  }
  static get random(){
    let h = 2 * Math.random() - 1;
    let r = Math.sqrt(1 - h * h);
    let phi = Math.random() * 2 * Math.PI;
    return new Vector3(r * Math.cos(phi), r * Math.sin(phi), h);
  }
}
class Matrix3 {
  constructor(v1, v2, v3){
    this.col1 = v1;
    this.col2 = v2;
    this.col3 = v3;
  }
  Multpl(x){
    return new Matrix3(this.col1.Multpl(x), this.col2.Multpl(x), this.col3.Multpl(x));
  }
  transform(v){
    return this.col1.Multpl(v.x).Add(this.col2.Multpl(v.y)).Add(this.col3.Multpl(v.z));
  }
  m(m){
    return new Matrix3(this.col1.m(m), this.col2.m(m), this.col3.m(m));
  }
  draw(camera, begin = Vector3.zero){
    camera.line3(begin, this.col1.Add(begin), "red");
    camera.line3(begin, this.col2.Add(begin), "green");
    camera.line3(begin, this.col3.Add(begin), "blue");
  }
  Rot(q){
    return new Matrix3(this.col1.Rot(q), this.col2.Rot(q), this.col3.Rot(q));
  }
  get inverse(){
    var a = this.col1.x, b = this.col2.x, c = this.col3.x, d = this.col1.y, e = this.col2.y, f = this.col3.y, g = this.col1.z, h = this.col2.z, k = this.col3.z, det = this.det;
    return new Matrix3(new Vector3(e*k-f*h,-d*k+f*g,d*h-e*g).Multpl(1 / det),new Vector3(-b*k+c*h,a*k-c*g,-a*h+b*g).Multpl(1 / det),new Vector3(b*f-c*e,-a*f+c*d,a*e-b*d).Multpl(1 / det));
  }
  get det(){
    var a = this.col1.x, b = this.col2.x, c = this.col3.x, d = this.col1.y, e = this.col2.y, f = this.col3.y, g = this.col1.z, h = this.col2.z, k = this.col3.z;
    return a*(e*k-f*h)-b*(d*k-f*g)+c*(d*h-e*g);
  }
  static Angles(ax, ay){
    return new Matrix3(new Vector3(Math.cos(ax)*Math.cos(ay), Math.sin(ax)*Math.cos(ay), Math.sin(ay)), new Vector3(-Math.sin(ax), Math.cos(ax), 0), new Vector3(-Math.sin(ay)*Math.cos(ax), -Math.sin(ay)*Math.sin(ax), Math.cos(ay)));
  }
  static Euler(x, y, z){
    return new Matrix3(new Vector3(Math.cos(y)*Math.cos(z),Math.cos(y)*Math.sin(z),-Math.sin(y)), new Vector3(Math.sin(x)*Math.sin(y)*Math.cos(z)-Math.cos(x)*Math.sin(z),Math.sin(x)*Math.sin(y)*Math.sin(z)+Math.cos(x)*Math.cos(z),Math.sin(x)*Math.cos(y)), new Vector3(Math.cos(x)*Math.sin(y)*Math.cos(z)+Math.sin(x)*Math.sin(z),Math.cos(x)*Math.sin(y)*Math.sin(z)-Math.sin(x)*Math.cos(z),Math.cos(x)*Math.cos(y)));
  }
  static Quat(q){
    return Matrix3.identity.Rot(q);
  }
  static get identity(){
    return new Matrix3(new Vector3(1,0,0), new Vector3(0,1,0), new Vector3(0,0,1));
  }
}
class Matrix2 {
  constructor(a = 1, b = 0, c = 0, d = 1){
    this.a = a;
    this.b = b;
    this.c = c;
    this.d = d;
  }

  Accum(m){
    this.a += m.a;
    this.b += m.b;
    this.c += m.c;
    this.d += m.d;
  }
  Decline(m){
    this.a -= m.a;
    this.b -= m.b;
    this.c -= m.c;
    this.d -= m.d;
  }

  Multpl(x){
    return new Matrix2(this.a * x, this.b * x, this.c * x, this.d * x);
  }

  Dist(v){
    return Math.hypot(this.x - v.x, this.y - v.y);
  }

  vProd(v){
    return new Vector2(this.a * v.x + this.b * v.y, this.c * v.x + this.d * v.y);
  }

  mProd(m){
    return new Matrix2(this.a*m.a+this.b*m.c, this.a*m.b+this.b*m.d, this.c*m.a+this.d*m.c, this.c*m.b+this.d*m.d);
  }

  rotate(phi){
    var c = Math.cos(phi), s = Math.sin(phi);
    var new_a = this.a * c + this.b * s;
    var new_b = this.b * c - this.a * s;
    var new_c = this.c * c + this.d * s;
    var new_d = this.d * c - this.c * s;
    this.a = new_a;
    this.b = new_b;
    this.c = new_c;
    this.d = new_d;
  }

  get det(){
    return this.a * this.d - this.b * this.c;
  }

  get inversed(){
    return new Matrix2(this.d, -this.b, -this.c, this.a).Multpl(1 / this.det);
  }

  static Angle(phi, r = 1){
    var c = Math.cos(phi) * r, s = Math.sin(phi) * r;
    return new Matrix2(c, -s, s, c);
  }
}
class Quaternion {
  constructor(a, v){
    this.a = a;
    this.v = v;
  }
  Add(q){
    return new Quaternion(this.a + q.a, this.v.Add(q.v));
  }
  Prod(q){
    return new Quaternion(this.a*q.a-this.v.x*q.v.x-this.v.y*q.v.y-this.v.z*q.v.z, new Vector3(this.a*q.v.x+this.v.x*q.a+this.v.y*q.v.z-this.v.z*q.v.y, this.a*q.v.y+this.v.y*q.a-this.v.x*q.v.z+this.v.z*q.v.x, this.a*q.v.z+this.v.z*q.a+this.v.x*q.v.y-this.v.y*q.v.x));
  }
  getEuler(){
    return new Vector3(Math.atan2(2*(this.a*this.v.x+this.v.y*this.v.z), 1-2*(this.v.x*this.v.x+this.v.y*this.v.y)), Math.asin(2*(this.a*this.v.y-this.v.x*this.v.z)), Math.atan2(2*(this.a*this.v.z+this.v.y*this.v.x), 1-2*(this.v.z*this.v.z+this.v.y*this.v.y)));
  }
  get conj(){
    return new Quaternion(this.a, this.v.Multpl(-1));
  }
  static rot(feta, vector){
    return new Quaternion(Math.cos(feta / 2), vector.normalized.Multpl(Math.sin(feta / 2)));
  }
}
class Sphere {
  constructor(v, r, emittance = false){
    this.v = v;
    this.r = r;
    this.lightEmittance = emittance;
  }
  Dist(v){
    return v.Add(this.v.Multpl(-1)).magnitude - this.r;
  }
  rayCast(x, v){
    var n = this.v.Add(x.Multpl(-1));
    var discr = this.r * this.r - n.magnitude2 + Math.pow(n.Dot(v), 2);
    if (discr >= 0){
      var r = n.Dot(v) - Math.sqrt(discr);
      return [x.Add(v.Multpl(r)), r];
    }
    return [null];
  }
  normal(x){
    return x.Add(this.v.Multpl(-1)).normalized;
  }
}

class Window {
  constructor(ctx, width, height, xmin = 0, xmax = 2 * (width / height), ymin = -1, ymax = 1){
    this.ctx = ctx;
    this.width = width;
    this.height = height;
    this.window = {Xmin: xmin, Xmax: xmax, Ymin: ymin, Ymax: ymax};
  }

  Clear(){
    this.ctx.clearRect(0, 0, this.width, this.height);
  }
  Pt_On(x, y, color = "black"){
    this.ctx.fillStyle = color;
    var v = this.Vector2ToPxl(x, y);
    this.ctx.fillRect(v.x - 1, v.y - 1, 3, 3);
  }
  drawLine(v, v1, color = "black", width = 2){
    this.ctx.beginPath();
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = width;
    this.ctx.lineJoin = "round";
    var v0 = this.Vector2ToPxl(v.x, v.y);
    var v2 = this.Vector2ToPxl(v1.x, v1.y);
    this.ctx.moveTo(v0.x, v0.y);
    this.ctx.lineTo(v2.x, v2.y);
    this.ctx.closePath();
    this.ctx.stroke();
  }
  drawPoly(p, center = Vector2.zero){
    this.ctx.beginPath();
    this.ctx.fillStyle = p.color;
    this.ctx.lineWidth = 2;
    this.ctx.lineJoin = "round";
    var v0 = this.Vector2ToPxl(center.x + p.verteces[0].x, center.y + p.verteces[0].y);
    this.ctx.moveTo(v0.x, v0.y);
    for (var i = 1; i < p.verteces.length; i++){
      v0 = this.Vector2ToPxl(center.x + p.verteces[i].x, center.y + p.verteces[i].y);
      this.ctx.lineTo(v0.x, v0.y);
    }
    this.ctx.closePath();
    this.ctx.fill();
  }
  drawCircle(m, r, stroke = "black", fill = null){
    this.ctx.beginPath();
    this.ctx.fillStyle = fill;
    this.ctx.strokeStyle = stroke;
    this.ctx.lineWidth = 2;
    var v = this.Vector2ToPxl(m.x, m.y);
    this.ctx.arc(v.x, v.y, r / (this.window.Xmax - this.window.Xmin) * this.width, 0, 2 * Math.PI);
    if (fill != null) this.ctx.fill();
    this.ctx.stroke();
  }
  drawText(x, y, str, color = "black", size = 15){
    this.ctx.font = size + "px Arial";
    this.ctx.textAlign = "center";
    this.ctx.fillStyle = color;
    var loc = this.Vector2ToPxl(x, y);
    this.ctx.fillText(str, loc.x, loc.y + 4);
  }
  drawRect(x, y, w, h, stroke = "black", fill = null){
    this.ctx.fillStyle = fill;
    this.ctx.strokeStyle = stroke;
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    var p = [this.Vector2ToPxl(x, y), this.Vector2ToPxl(x + w, y), this.Vector2ToPxl(x + w, y + h), this.Vector2ToPxl(x, y + h)];
    this.ctx.moveTo(p[3].x, p[3].y);
    for (var i = 0; i < 4; i++) this.ctx.lineTo(p[i].x, p[i].y);
    if (fill != null) this.ctx.fill();
    this.ctx.stroke();
  }
  Vector2FromPxl(x, y){
    return new Vector2(x/this.width*(this.window.Xmax - this.window.Xmin)+this.window.Xmin, y/this.height*(this.window.Ymin - this.window.Ymax)+this.window.Ymax);
  }
  Vector2ToPxl(x, y){
    return new Vector2(Math.floor((x - this.window.Xmin)/(this.window.Xmax - this.window.Xmin)*this.width), Math.floor((y - this.window.Ymax)/(this.window.Ymin - this.window.Ymax)*this.height));
  }
  zSquare(){
    if (this.width / this.height > (this.window.Xmax - this.window.Xmin) / (this.window.Ymax - this.window.Ymin)){
      var dx = this.width / this.height * (this.window.Ymax - this.window.Ymin) - (this.window.Xmax - this.window.Xmin);
      this.window.Xmax += 0.5 * dx;
      this.window.Xmin -= 0.5 * dx;
    } else {
      var dy = this.height / this.width * (this.window.Xmax - this.window.Xmin) - (this.window.Ymax - this.window.Ymin);
      this.window.Ymax += 0.5 * dy;
      this.window.Ymin -= 0.5 * dy;
    }
  }
}
class Camera {
  constructor(ctx, width, height){
    this.window = new Window(ctx, width, height, -width / height, width / height, -1, 1);
    this.WorldToLocal;
    this.LocalToWorld;
    this.cameraPos = new Vector3(15, 15, 21.21).Multpl(1.5);
    this.screendist = 42;
    this.lookAt();
  }

  Clear(){
    this.window.Clear();
  }

  Vector3ToPxl(v){
    var vect = this.WorldToLocal.transform(v.Sub(this.cameraPos));
    return this.window.Vector2ToPxl(vect.x * this.screendist / vect.z, vect.y * this.screendist / vect.z);
  }

  Vector3FromPxl(x, y){
    var v = this.window.Vector2FromPxl(x, y);
    return new Vector3(v.x, v.y, this.screendist).m(this.LocalToWorld).normalized;
  }

  lookAt(v = Vector3.zero, up = new Vector3(0, 0, 1)){
    var z = v.Sub(this.cameraPos).normalized;
    var x = z.Crs(up).normalized;
    var y = x.Crs(z);
    this.WorldToLocal = new Matrix3(x, y, z).inverse;
    this.LocalToWorld = new Matrix3(x, y, z);
  }

  line3(v1, v2, color = "black"){
    this.window.ctx.beginPath();
    this.window.ctx.strokeStyle = color;
    this.window.ctx.lineWidth = 1;
    this.window.ctx.lineJoin = "round";
    var begin = this.Vector3ToPxl(v1);
    var eind = this.Vector3ToPxl(v2);
    this.window.ctx.moveTo(begin.x, begin.y);
    this.window.ctx.lineTo(eind.x, eind.y);
    this.window.ctx.closePath();
    this.window.ctx.stroke();
  }

  point3(v, color = "black", r = 0.02){
    if (v.Sub(this.cameraPos).Dot(this.LocalToWorld.col3) < 0) return;
    var vect = this.Vector3ToPxl(v);
    this.window.ctx.beginPath();
    this.window.ctx.fillStyle = color;
    this.window.ctx.arc(vect.x, vect.y, 0.5 * this.window.height * r * this.screendist / Math.sqrt(v.Sub(this.cameraPos).magnitude2 - r * r), 0, 2 * Math.PI);
    this.window.ctx.fill();
  }

  setupTrackball(radius, position = Vector3.zero, func = null){
    let camera = this;
    this.lastPosition = null;
    this.trackball = new Sphere(position, radius);
    this.lookAt(position);
    this.trackball_event_function = func;
    this.window.ctx.canvas.onmousedown = function (e){
      camera.lastPosition = camera.trackball.rayCast(camera.cameraPos, camera.Vector3FromPxl(e.pageX, e.pageY))[0];
    }
    this.window.ctx.canvas.onmousemove = function (e){
      if (camera.lastPosition != null){
        let new_position = camera.trackball.rayCast(camera.cameraPos, camera.Vector3FromPxl(e.pageX, e.pageY))[0];
        let angle;
        if (new_position == null || (angle = new_position.Dist(camera.lastPosition)) == 0) return;
        let q = Quaternion.rot(angle, new_position.Crs(camera.lastPosition));
        let rel = camera.cameraPos.Sub(camera.trackball.v).Rot(q);
        camera.cameraPos = camera.trackball.v.Add(rel);
        camera.lookAt(camera.trackball.v);
        camera.lastPosition = camera.trackball.rayCast(camera.cameraPos, camera.Vector3FromPxl(e.pageX, e.pageY))[0];
        camera.trackball_event_function.call();
      }
    }
    this.window.ctx.canvas.onmouseup = function () { camera.lastPosition = null; }
    this.window.ctx.canvas.onmouseleave = function () { camera.lastPosition = null; }
    this.window.ctx.canvas.onwheel = function (e) {
      var factor = 1 + 0.1 * Math.sign(e.deltaY);
      camera.cameraPos = camera.trackball.v.Add(camera.cameraPos.Sub(camera.trackball.v).Multpl(factor));
      camera.trackball_event_function.call();
      camera.lastPosition = null;
    }
  }
}
