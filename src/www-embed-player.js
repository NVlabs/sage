(function(){/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
'use strict';var l;function aa(a){var b=0;return function(){return b<a.length?{done:!1,value:a[b++]}:{done:!0}}}
var ba="function"==typeof Object.defineProperties?Object.defineProperty:function(a,b,c){if(a==Array.prototype||a==Object.prototype)return a;a[b]=c.value;return a};
function da(a){a=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global];for(var b=0;b<a.length;++b){var c=a[b];if(c&&c.Math==Math)return c}throw Error("Cannot find global object");}
var fa=da(this);function n(a,b){if(b)a:{var c=fa;a=a.split(".");for(var d=0;d<a.length-1;d++){var e=a[d];if(!(e in c))break a;c=c[e]}a=a[a.length-1];d=c[a];b=b(d);b!=d&&null!=b&&ba(c,a,{configurable:!0,writable:!0,value:b})}}
n("Symbol",function(a){function b(f){if(this instanceof b)throw new TypeError("Symbol is not a constructor");return new c(d+(f||"")+"_"+e++,f)}
function c(f,g){this.h=f;ba(this,"description",{configurable:!0,writable:!0,value:g})}
if(a)return a;c.prototype.toString=function(){return this.h};
var d="jscomp_symbol_"+(1E9*Math.random()>>>0)+"_",e=0;return b});
n("Symbol.iterator",function(a){if(a)return a;a=Symbol("Symbol.iterator");for(var b="Array Int8Array Uint8Array Uint8ClampedArray Int16Array Uint16Array Int32Array Uint32Array Float32Array Float64Array".split(" "),c=0;c<b.length;c++){var d=fa[b[c]];"function"===typeof d&&"function"!=typeof d.prototype[a]&&ba(d.prototype,a,{configurable:!0,writable:!0,value:function(){return ha(aa(this))}})}return a});
function ha(a){a={next:a};a[Symbol.iterator]=function(){return this};
return a}
function p(a){var b="undefined"!=typeof Symbol&&Symbol.iterator&&a[Symbol.iterator];return b?b.call(a):{next:aa(a)}}
function ia(a){if(!(a instanceof Array)){a=p(a);for(var b,c=[];!(b=a.next()).done;)c.push(b.value);a=c}return a}
function ja(a,b){return Object.prototype.hasOwnProperty.call(a,b)}
var ka="function"==typeof Object.assign?Object.assign:function(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(d)for(var e in d)ja(d,e)&&(a[e]=d[e])}return a};
n("Object.assign",function(a){return a||ka});
var la="function"==typeof Object.create?Object.create:function(a){function b(){}
b.prototype=a;return new b},oa=function(){function a(){function c(){}
new c;Reflect.construct(c,[],function(){});
return new c instanceof c}
if("undefined"!=typeof Reflect&&Reflect.construct){if(a())return Reflect.construct;var b=Reflect.construct;return function(c,d,e){c=b(c,d);e&&Reflect.setPrototypeOf(c,e.prototype);return c}}return function(c,d,e){void 0===e&&(e=c);
e=la(e.prototype||Object.prototype);return Function.prototype.apply.call(c,e,d)||e}}(),qa;
if("function"==typeof Object.setPrototypeOf)qa=Object.setPrototypeOf;else{var ra;a:{var sa={a:!0},ta={};try{ta.__proto__=sa;ra=ta.a;break a}catch(a){}ra=!1}qa=ra?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var ua=qa;
function u(a,b){a.prototype=la(b.prototype);a.prototype.constructor=a;if(ua)ua(a,b);else for(var c in b)if("prototype"!=c)if(Object.defineProperties){var d=Object.getOwnPropertyDescriptor(b,c);d&&Object.defineProperty(a,c,d)}else a[c]=b[c];a.X=b.prototype}
function va(){this.v=!1;this.l=null;this.i=void 0;this.h=1;this.m=this.o=0;this.s=this.j=null}
function wa(a){if(a.v)throw new TypeError("Generator is already running");a.v=!0}
va.prototype.M=function(a){this.i=a};
function xa(a,b){a.j={Yb:b,fc:!0};a.h=a.o||a.m}
va.prototype.return=function(a){this.j={return:a};this.h=this.m};
function v(a,b,c){a.h=c;return{value:b}}
va.prototype.u=function(a){this.h=a};
function ya(a,b,c){a.o=b;void 0!=c&&(a.m=c)}
function za(a,b){a.h=b;a.o=0}
function Aa(a){a.o=0;var b=a.j.Yb;a.j=null;return b}
function Ba(a){a.s=[a.j];a.o=0;a.m=0}
function Ca(a){var b=a.s.splice(0)[0];(b=a.j=a.j||b)?b.fc?a.h=a.o||a.m:void 0!=b.u&&a.m<b.u?(a.h=b.u,a.j=null):a.h=a.m:a.h=0}
function Da(a){this.h=new va;this.i=a}
function Ea(a,b){wa(a.h);var c=a.h.l;if(c)return Fa(a,"return"in c?c["return"]:function(d){return{value:d,done:!0}},b,a.h.return);
a.h.return(b);return Ga(a)}
function Fa(a,b,c,d){try{var e=b.call(a.h.l,c);if(!(e instanceof Object))throw new TypeError("Iterator result "+e+" is not an object");if(!e.done)return a.h.v=!1,e;var f=e.value}catch(g){return a.h.l=null,xa(a.h,g),Ga(a)}a.h.l=null;d.call(a.h,f);return Ga(a)}
function Ga(a){for(;a.h.h;)try{var b=a.i(a.h);if(b)return a.h.v=!1,{value:b.value,done:!1}}catch(c){a.h.i=void 0,xa(a.h,c)}a.h.v=!1;if(a.h.j){b=a.h.j;a.h.j=null;if(b.fc)throw b.Yb;return{value:b.return,done:!0}}return{value:void 0,done:!0}}
function Ha(a){this.next=function(b){wa(a.h);a.h.l?b=Fa(a,a.h.l.next,b,a.h.M):(a.h.M(b),b=Ga(a));return b};
this.throw=function(b){wa(a.h);a.h.l?b=Fa(a,a.h.l["throw"],b,a.h.M):(xa(a.h,b),b=Ga(a));return b};
this.return=function(b){return Ea(a,b)};
this[Symbol.iterator]=function(){return this}}
function Ia(a){function b(d){return a.next(d)}
function c(d){return a.throw(d)}
return new Promise(function(d,e){function f(g){g.done?d(g.value):Promise.resolve(g.value).then(b,c).then(f,e)}
f(a.next())})}
function x(a){return Ia(new Ha(new Da(a)))}
function Ja(){for(var a=Number(this),b=[],c=a;c<arguments.length;c++)b[c-a]=arguments[c];return b}
n("Reflect",function(a){return a?a:{}});
n("Reflect.construct",function(){return oa});
n("Reflect.setPrototypeOf",function(a){return a?a:ua?function(b,c){try{return ua(b,c),!0}catch(d){return!1}}:null});
n("Promise",function(a){function b(g){this.h=0;this.j=void 0;this.i=[];this.v=!1;var h=this.l();try{g(h.resolve,h.reject)}catch(k){h.reject(k)}}
function c(){this.h=null}
function d(g){return g instanceof b?g:new b(function(h){h(g)})}
if(a)return a;c.prototype.i=function(g){if(null==this.h){this.h=[];var h=this;this.j(function(){h.m()})}this.h.push(g)};
var e=fa.setTimeout;c.prototype.j=function(g){e(g,0)};
c.prototype.m=function(){for(;this.h&&this.h.length;){var g=this.h;this.h=[];for(var h=0;h<g.length;++h){var k=g[h];g[h]=null;try{k()}catch(m){this.l(m)}}}this.h=null};
c.prototype.l=function(g){this.j(function(){throw g;})};
b.prototype.l=function(){function g(m){return function(q){k||(k=!0,m.call(h,q))}}
var h=this,k=!1;return{resolve:g(this.K),reject:g(this.m)}};
b.prototype.K=function(g){if(g===this)this.m(new TypeError("A Promise cannot resolve to itself"));else if(g instanceof b)this.P(g);else{a:switch(typeof g){case "object":var h=null!=g;break a;case "function":h=!0;break a;default:h=!1}h?this.I(g):this.o(g)}};
b.prototype.I=function(g){var h=void 0;try{h=g.then}catch(k){this.m(k);return}"function"==typeof h?this.S(h,g):this.o(g)};
b.prototype.m=function(g){this.M(2,g)};
b.prototype.o=function(g){this.M(1,g)};
b.prototype.M=function(g,h){if(0!=this.h)throw Error("Cannot settle("+g+", "+h+"): Promise already settled in state"+this.h);this.h=g;this.j=h;2===this.h&&this.O();this.s()};
b.prototype.O=function(){var g=this;e(function(){if(g.F()){var h=fa.console;"undefined"!==typeof h&&h.error(g.j)}},1)};
b.prototype.F=function(){if(this.v)return!1;var g=fa.CustomEvent,h=fa.Event,k=fa.dispatchEvent;if("undefined"===typeof k)return!0;"function"===typeof g?g=new g("unhandledrejection",{cancelable:!0}):"function"===typeof h?g=new h("unhandledrejection",{cancelable:!0}):(g=fa.document.createEvent("CustomEvent"),g.initCustomEvent("unhandledrejection",!1,!0,g));g.promise=this;g.reason=this.j;return k(g)};
b.prototype.s=function(){if(null!=this.i){for(var g=0;g<this.i.length;++g)f.i(this.i[g]);this.i=null}};
var f=new c;b.prototype.P=function(g){var h=this.l();g.eb(h.resolve,h.reject)};
b.prototype.S=function(g,h){var k=this.l();try{g.call(h,k.resolve,k.reject)}catch(m){k.reject(m)}};
b.prototype.then=function(g,h){function k(w,t){return"function"==typeof w?function(y){try{m(w(y))}catch(E){q(E)}}:t}
var m,q,r=new b(function(w,t){m=w;q=t});
this.eb(k(g,m),k(h,q));return r};
b.prototype.catch=function(g){return this.then(void 0,g)};
b.prototype.eb=function(g,h){function k(){switch(m.h){case 1:g(m.j);break;case 2:h(m.j);break;default:throw Error("Unexpected state: "+m.h);}}
var m=this;null==this.i?f.i(k):this.i.push(k);this.v=!0};
b.resolve=d;b.reject=function(g){return new b(function(h,k){k(g)})};
b.race=function(g){return new b(function(h,k){for(var m=p(g),q=m.next();!q.done;q=m.next())d(q.value).eb(h,k)})};
b.all=function(g){var h=p(g),k=h.next();return k.done?d([]):new b(function(m,q){function r(y){return function(E){w[y]=E;t--;0==t&&m(w)}}
var w=[],t=0;do w.push(void 0),t++,d(k.value).eb(r(w.length-1),q),k=h.next();while(!k.done)})};
return b});
n("WeakMap",function(a){function b(k){this.h=(h+=Math.random()+1).toString();if(k){k=p(k);for(var m;!(m=k.next()).done;)m=m.value,this.set(m[0],m[1])}}
function c(){}
function d(k){var m=typeof k;return"object"===m&&null!==k||"function"===m}
function e(k){if(!ja(k,g)){var m=new c;ba(k,g,{value:m})}}
function f(k){var m=Object[k];m&&(Object[k]=function(q){if(q instanceof c)return q;Object.isExtensible(q)&&e(q);return m(q)})}
if(function(){if(!a||!Object.seal)return!1;try{var k=Object.seal({}),m=Object.seal({}),q=new a([[k,2],[m,3]]);if(2!=q.get(k)||3!=q.get(m))return!1;q.delete(k);q.set(m,4);return!q.has(k)&&4==q.get(m)}catch(r){return!1}}())return a;
var g="$jscomp_hidden_"+Math.random();f("freeze");f("preventExtensions");f("seal");var h=0;b.prototype.set=function(k,m){if(!d(k))throw Error("Invalid WeakMap key");e(k);if(!ja(k,g))throw Error("WeakMap key fail: "+k);k[g][this.h]=m;return this};
b.prototype.get=function(k){return d(k)&&ja(k,g)?k[g][this.h]:void 0};
b.prototype.has=function(k){return d(k)&&ja(k,g)&&ja(k[g],this.h)};
b.prototype.delete=function(k){return d(k)&&ja(k,g)&&ja(k[g],this.h)?delete k[g][this.h]:!1};
return b});
n("Map",function(a){function b(){var h={};return h.previous=h.next=h.head=h}
function c(h,k){var m=h.h;return ha(function(){if(m){for(;m.head!=h.h;)m=m.previous;for(;m.next!=m.head;)return m=m.next,{done:!1,value:k(m)};m=null}return{done:!0,value:void 0}})}
function d(h,k){var m=k&&typeof k;"object"==m||"function"==m?f.has(k)?m=f.get(k):(m=""+ ++g,f.set(k,m)):m="p_"+k;var q=h.data_[m];if(q&&ja(h.data_,m))for(h=0;h<q.length;h++){var r=q[h];if(k!==k&&r.key!==r.key||k===r.key)return{id:m,list:q,index:h,entry:r}}return{id:m,list:q,index:-1,entry:void 0}}
function e(h){this.data_={};this.h=b();this.size=0;if(h){h=p(h);for(var k;!(k=h.next()).done;)k=k.value,this.set(k[0],k[1])}}
if(function(){if(!a||"function"!=typeof a||!a.prototype.entries||"function"!=typeof Object.seal)return!1;try{var h=Object.seal({x:4}),k=new a(p([[h,"s"]]));if("s"!=k.get(h)||1!=k.size||k.get({x:4})||k.set({x:4},"t")!=k||2!=k.size)return!1;var m=k.entries(),q=m.next();if(q.done||q.value[0]!=h||"s"!=q.value[1])return!1;q=m.next();return q.done||4!=q.value[0].x||"t"!=q.value[1]||!m.next().done?!1:!0}catch(r){return!1}}())return a;
var f=new WeakMap;e.prototype.set=function(h,k){h=0===h?0:h;var m=d(this,h);m.list||(m.list=this.data_[m.id]=[]);m.entry?m.entry.value=k:(m.entry={next:this.h,previous:this.h.previous,head:this.h,key:h,value:k},m.list.push(m.entry),this.h.previous.next=m.entry,this.h.previous=m.entry,this.size++);return this};
e.prototype.delete=function(h){h=d(this,h);return h.entry&&h.list?(h.list.splice(h.index,1),h.list.length||delete this.data_[h.id],h.entry.previous.next=h.entry.next,h.entry.next.previous=h.entry.previous,h.entry.head=null,this.size--,!0):!1};
e.prototype.clear=function(){this.data_={};this.h=this.h.previous=b();this.size=0};
e.prototype.has=function(h){return!!d(this,h).entry};
e.prototype.get=function(h){return(h=d(this,h).entry)&&h.value};
e.prototype.entries=function(){return c(this,function(h){return[h.key,h.value]})};
e.prototype.keys=function(){return c(this,function(h){return h.key})};
e.prototype.values=function(){return c(this,function(h){return h.value})};
e.prototype.forEach=function(h,k){for(var m=this.entries(),q;!(q=m.next()).done;)q=q.value,h.call(k,q[1],q[0],this)};
e.prototype[Symbol.iterator]=e.prototype.entries;var g=0;return e});
function Ka(a,b,c){if(null==a)throw new TypeError("The 'this' value for String.prototype."+c+" must not be null or undefined");if(b instanceof RegExp)throw new TypeError("First argument to String.prototype."+c+" must not be a regular expression");return a+""}
n("String.prototype.endsWith",function(a){return a?a:function(b,c){var d=Ka(this,b,"endsWith");b+="";void 0===c&&(c=d.length);c=Math.max(0,Math.min(c|0,d.length));for(var e=b.length;0<e&&0<c;)if(d[--c]!=b[--e])return!1;return 0>=e}});
n("Array.prototype.find",function(a){return a?a:function(b,c){a:{var d=this;d instanceof String&&(d=String(d));for(var e=d.length,f=0;f<e;f++){var g=d[f];if(b.call(c,g,f,d)){b=g;break a}}b=void 0}return b}});
n("String.prototype.startsWith",function(a){return a?a:function(b,c){var d=Ka(this,b,"startsWith");b+="";var e=d.length,f=b.length;c=Math.max(0,Math.min(c|0,d.length));for(var g=0;g<f&&c<e;)if(d[c++]!=b[g++])return!1;return g>=f}});
n("Number.isFinite",function(a){return a?a:function(b){return"number"!==typeof b?!1:!isNaN(b)&&Infinity!==b&&-Infinity!==b}});
n("Number.isInteger",function(a){return a?a:function(b){return Number.isFinite(b)?b===Math.floor(b):!1}});
n("Number.MAX_SAFE_INTEGER",function(){return 9007199254740991});
function La(a,b){a instanceof String&&(a+="");var c=0,d=!1,e={next:function(){if(!d&&c<a.length){var f=c++;return{value:b(f,a[f]),done:!1}}d=!0;return{done:!0,value:void 0}}};
e[Symbol.iterator]=function(){return e};
return e}
n("Array.prototype.entries",function(a){return a?a:function(){return La(this,function(b,c){return[b,c]})}});
n("Array.from",function(a){return a?a:function(b,c,d){c=null!=c?c:function(h){return h};
var e=[],f="undefined"!=typeof Symbol&&Symbol.iterator&&b[Symbol.iterator];if("function"==typeof f){b=f.call(b);for(var g=0;!(f=b.next()).done;)e.push(c.call(d,f.value,g++))}else for(f=b.length,g=0;g<f;g++)e.push(c.call(d,b[g],g));return e}});
n("Array.prototype.keys",function(a){return a?a:function(){return La(this,function(b){return b})}});
n("Array.prototype.values",function(a){return a?a:function(){return La(this,function(b,c){return c})}});
n("Number.isNaN",function(a){return a?a:function(b){return"number"===typeof b&&isNaN(b)}});
n("Object.setPrototypeOf",function(a){return a||ua});
n("Set",function(a){function b(c){this.h=new Map;if(c){c=p(c);for(var d;!(d=c.next()).done;)this.add(d.value)}this.size=this.h.size}
if(function(){if(!a||"function"!=typeof a||!a.prototype.entries||"function"!=typeof Object.seal)return!1;try{var c=Object.seal({x:4}),d=new a(p([c]));if(!d.has(c)||1!=d.size||d.add(c)!=d||1!=d.size||d.add({x:4})!=d||2!=d.size)return!1;var e=d.entries(),f=e.next();if(f.done||f.value[0]!=c||f.value[1]!=c)return!1;f=e.next();return f.done||f.value[0]==c||4!=f.value[0].x||f.value[1]!=f.value[0]?!1:e.next().done}catch(g){return!1}}())return a;
b.prototype.add=function(c){c=0===c?0:c;this.h.set(c,c);this.size=this.h.size;return this};
b.prototype.delete=function(c){c=this.h.delete(c);this.size=this.h.size;return c};
b.prototype.clear=function(){this.h.clear();this.size=0};
b.prototype.has=function(c){return this.h.has(c)};
b.prototype.entries=function(){return this.h.entries()};
b.prototype.values=function(){return this.h.values()};
b.prototype.keys=b.prototype.values;b.prototype[Symbol.iterator]=b.prototype.values;b.prototype.forEach=function(c,d){var e=this;this.h.forEach(function(f){return c.call(d,f,f,e)})};
return b});
n("Object.entries",function(a){return a?a:function(b){var c=[],d;for(d in b)ja(b,d)&&c.push([d,b[d]]);return c}});
n("Object.is",function(a){return a?a:function(b,c){return b===c?0!==b||1/b===1/c:b!==b&&c!==c}});
n("Array.prototype.includes",function(a){return a?a:function(b,c){var d=this;d instanceof String&&(d=String(d));var e=d.length;c=c||0;for(0>c&&(c=Math.max(c+e,0));c<e;c++){var f=d[c];if(f===b||Object.is(f,b))return!0}return!1}});
n("String.prototype.includes",function(a){return a?a:function(b,c){return-1!==Ka(this,b,"includes").indexOf(b,c||0)}});
n("globalThis",function(a){return a||fa});
n("Object.values",function(a){return a?a:function(b){var c=[],d;for(d in b)ja(b,d)&&c.push(b[d]);return c}});
var Ma=Ma||{},z=this||self;function A(a,b,c){a=a.split(".");c=c||z;a[0]in c||"undefined"==typeof c.execScript||c.execScript("var "+a[0]);for(var d;a.length&&(d=a.shift());)a.length||void 0===b?c[d]&&c[d]!==Object.prototype[d]?c=c[d]:c=c[d]={}:c[d]=b}
function B(a,b){a=a.split(".");b=b||z;for(var c=0;c<a.length;c++)if(b=b[a[c]],null==b)return null;return b}
function Na(a){a.zb=void 0;a.getInstance=function(){return a.zb?a.zb:a.zb=new a}}
function Oa(a){var b=typeof a;return"object"!=b?b:a?Array.isArray(a)?"array":b:"null"}
function Pa(a){var b=Oa(a);return"array"==b||"object"==b&&"number"==typeof a.length}
function Qa(a){var b=typeof a;return"object"==b&&null!=a||"function"==b}
function Ra(a){return Object.prototype.hasOwnProperty.call(a,Sa)&&a[Sa]||(a[Sa]=++Ta)}
var Sa="closure_uid_"+(1E9*Math.random()>>>0),Ta=0;function Ua(a,b,c){return a.call.apply(a.bind,arguments)}
function Wa(a,b,c){if(!a)throw Error();if(2<arguments.length){var d=Array.prototype.slice.call(arguments,2);return function(){var e=Array.prototype.slice.call(arguments);Array.prototype.unshift.apply(e,d);return a.apply(b,e)}}return function(){return a.apply(b,arguments)}}
function Xa(a,b,c){Function.prototype.bind&&-1!=Function.prototype.bind.toString().indexOf("native code")?Xa=Ua:Xa=Wa;return Xa.apply(null,arguments)}
function Ya(a,b){var c=Array.prototype.slice.call(arguments,1);return function(){var d=c.slice();d.push.apply(d,arguments);return a.apply(this,d)}}
function Za(a,b){function c(){}
c.prototype=b.prototype;a.X=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.lr=function(d,e,f){for(var g=Array(arguments.length-2),h=2;h<arguments.length;h++)g[h-2]=arguments[h];return b.prototype[e].apply(d,g)}}
function $a(a){return a}
;function ab(a,b){if(Error.captureStackTrace)Error.captureStackTrace(this,ab);else{var c=Error().stack;c&&(this.stack=c)}a&&(this.message=String(a));void 0!==b&&(this.cause=b)}
Za(ab,Error);ab.prototype.name="CustomError";function bb(a){a=a.url;var b=/[?&]dsh=1(&|$)/.test(a);this.j=!b&&/[?&]ae=1(&|$)/.test(a);this.l=!b&&/[?&]ae=2(&|$)/.test(a);if((this.h=/[?&]adurl=([^&]*)/.exec(a))&&this.h[1]){try{var c=decodeURIComponent(this.h[1])}catch(d){c=null}this.i=c}}
;function cb(){}
function db(a){var b=!1,c;return function(){b||(c=a(),b=!0);return c}}
;var eb=Array.prototype.indexOf?function(a,b){return Array.prototype.indexOf.call(a,b,void 0)}:function(a,b){if("string"===typeof a)return"string"!==typeof b||1!=b.length?-1:a.indexOf(b,0);
for(var c=0;c<a.length;c++)if(c in a&&a[c]===b)return c;return-1},fb=Array.prototype.forEach?function(a,b,c){Array.prototype.forEach.call(a,b,c)}:function(a,b,c){for(var d=a.length,e="string"===typeof a?a.split(""):a,f=0;f<d;f++)f in e&&b.call(c,e[f],f,a)},gb=Array.prototype.filter?function(a,b){return Array.prototype.filter.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=[],e=0,f="string"===typeof a?a.split(""):a,g=0;g<c;g++)if(g in f){var h=f[g];
b.call(void 0,h,g,a)&&(d[e++]=h)}return d},hb=Array.prototype.map?function(a,b){return Array.prototype.map.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=Array(c),e="string"===typeof a?a.split(""):a,f=0;f<c;f++)f in e&&(d[f]=b.call(void 0,e[f],f,a));
return d},ib=Array.prototype.reduce?function(a,b,c){return Array.prototype.reduce.call(a,b,c)}:function(a,b,c){var d=c;
fb(a,function(e,f){d=b.call(void 0,d,e,f,a)});
return d};
function jb(a,b){a:{for(var c=a.length,d="string"===typeof a?a.split(""):a,e=0;e<c;e++)if(e in d&&b.call(void 0,d[e],e,a)){b=e;break a}b=-1}return 0>b?null:"string"===typeof a?a.charAt(b):a[b]}
function kb(a,b){b=eb(a,b);var c;(c=0<=b)&&Array.prototype.splice.call(a,b,1);return c}
function lb(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(Pa(d)){var e=a.length||0,f=d.length||0;a.length=e+f;for(var g=0;g<f;g++)a[e+g]=d[g]}else a.push(d)}}
;function mb(a,b){for(var c in a)b.call(void 0,a[c],c,a)}
function nb(a){var b=ob,c;for(c in b)if(a.call(void 0,b[c],c,b))return c}
function pb(a){for(var b in a)return!1;return!0}
function qb(a,b){if(null!==a&&b in a)throw Error('The object already contains the key "'+b+'"');a[b]=!0}
function sb(a){return null!==a&&"privembed"in a?a.privembed:!1}
function tb(a,b){for(var c in a)if(!(c in b)||a[c]!==b[c])return!1;for(var d in b)if(!(d in a))return!1;return!0}
function ub(a){var b={},c;for(c in a)b[c]=a[c];return b}
function vb(a){if(!a||"object"!==typeof a)return a;if("function"===typeof a.clone)return a.clone();if("undefined"!==typeof Map&&a instanceof Map)return new Map(a);if("undefined"!==typeof Set&&a instanceof Set)return new Set(a);if(a instanceof Date)return new Date(a.getTime());var b=Array.isArray(a)?[]:"function"!==typeof ArrayBuffer||"function"!==typeof ArrayBuffer.isView||!ArrayBuffer.isView(a)||a instanceof DataView?{}:new a.constructor(a.length),c;for(c in a)b[c]=vb(a[c]);return b}
var wb="constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");function xb(a,b){for(var c,d,e=1;e<arguments.length;e++){d=arguments[e];for(c in d)a[c]=d[c];for(var f=0;f<wb.length;f++)c=wb[f],Object.prototype.hasOwnProperty.call(d,c)&&(a[c]=d[c])}}
;var yb;function zb(){if(void 0===yb){var a=null,b=z.trustedTypes;if(b&&b.createPolicy){try{a=b.createPolicy("goog#html",{createHTML:$a,createScript:$a,createScriptURL:$a})}catch(c){z.console&&z.console.error(c.message)}yb=a}else yb=a}return yb}
;function Ab(a,b){this.j=a===Bb&&b||""}
Ab.prototype.i=!0;Ab.prototype.h=function(){return this.j};
function Cb(a){return new Ab(Bb,a)}
var Bb={};Cb("");var Db={};function Eb(a){this.j=Db===Db?a:"";this.i=!0}
Eb.prototype.toString=function(){return this.j.toString()};
Eb.prototype.h=function(){return this.j.toString()};function Fb(a,b){this.j=b===Gb?a:""}
Fb.prototype.toString=function(){return this.j+""};
Fb.prototype.i=!0;Fb.prototype.h=function(){return this.j.toString()};
function Hb(a){if(a instanceof Fb&&a.constructor===Fb)return a.j;Oa(a);return"type_error:TrustedResourceUrl"}
var Gb={};function Ib(a){var b=zb();a=b?b.createScriptURL(a):a;return new Fb(a,Gb)}
;var Jb=String.prototype.trim?function(a){return a.trim()}:function(a){return/^[\s\xa0]*([\s\S]*?)[\s\xa0]*$/.exec(a)[1]};
function Kb(a,b){return a<b?-1:a>b?1:0}
;function Nb(a,b){this.j=b===Ob?a:""}
Nb.prototype.toString=function(){return this.j.toString()};
Nb.prototype.i=!0;Nb.prototype.h=function(){return this.j.toString()};
function Pb(a){if(a instanceof Nb&&a.constructor===Nb)return a.j;Oa(a);return"type_error:SafeUrl"}
var Qb=/^(?:(?:https?|mailto|ftp):|[^:/?#]*(?:[/?#]|$))/i,Ob={},Rb=new Nb("about:invalid#zClosurez",Ob);function Sb(){var a=z.navigator;return a&&(a=a.userAgent)?a:""}
function C(a){return-1!=Sb().indexOf(a)}
;function Tb(){return C("Trident")||C("MSIE")}
function Ub(){return C("Firefox")||C("FxiOS")}
function Vb(){return C("Safari")&&!(Wb()||C("Coast")||C("Opera")||C("Edge")||C("Edg/")||C("OPR")||Ub()||C("Silk")||C("Android"))}
function Wb(){return(C("Chrome")||C("CriOS"))&&!C("Edge")||C("Silk")}
function Xb(){return C("Android")&&!(Wb()||Ub()||C("Opera")||C("Silk"))}
function Yb(a){var b={};a.forEach(function(c){b[c[0]]=c[1]});
return function(c){return b[c.find(function(d){return d in b})]||""}}
function Zb(a){var b=Sb();if("Internet Explorer"===a){if(Tb())if((a=/rv: *([\d\.]*)/.exec(b))&&a[1])b=a[1];else{a="";var c=/MSIE +([\d\.]+)/.exec(b);if(c&&c[1])if(b=/Trident\/(\d.\d)/.exec(b),"7.0"==c[1])if(b&&b[1])switch(b[1]){case "4.0":a="8.0";break;case "5.0":a="9.0";break;case "6.0":a="10.0";break;case "7.0":a="11.0"}else a="7.0";else a=c[1];b=a}else b="";return b}var d=RegExp("([A-Z][\\w ]+)/([^\\s]+)\\s*(?:\\((.*?)\\))?","g");c=[];for(var e;e=d.exec(b);)c.push([e[1],e[2],e[3]||void 0]);b=Yb(c);
switch(a){case "Opera":if(C("Opera"))return b(["Version","Opera"]);if(C("OPR"))return b(["OPR"]);break;case "Microsoft Edge":if(C("Edge"))return b(["Edge"]);if(C("Edg/"))return b(["Edg"]);break;case "Chromium":if(Wb())return b(["Chrome","CriOS","HeadlessChrome"])}return"Firefox"===a&&Ub()||"Safari"===a&&Vb()||"Android Browser"===a&&Xb()||"Silk"===a&&C("Silk")?(b=c[2])&&b[1]||"":""}
function $b(a){a=Zb(a);if(""===a)return NaN;a=a.split(".");return 0===a.length?NaN:Number(a[0])}
;var ac={};function dc(a){this.j=ac===ac?a:"";this.i=!0}
dc.prototype.h=function(){return this.j.toString()};
dc.prototype.toString=function(){return this.j.toString()};function ec(a,b){b instanceof Nb||b instanceof Nb||(b="object"==typeof b&&b.i?b.h():String(b),Qb.test(b)||(b="about:invalid#zClosurez"),b=new Nb(b,Ob));a.href=Pb(b)}
function fc(a,b){a.rel="stylesheet";a.href=Hb(b).toString();(b=gc('style[nonce],link[rel="stylesheet"][nonce]',a.ownerDocument&&a.ownerDocument.defaultView))&&a.setAttribute("nonce",b)}
function hc(){return gc("script[nonce]")}
var ic=/^[\w+/_-]+[=]{0,2}$/;function gc(a,b){b=(b||z).document;return b.querySelector?(a=b.querySelector(a))&&(a=a.nonce||a.getAttribute("nonce"))&&ic.test(a)?a:"":""}
;function jc(a){for(var b=0,c=0;c<a.length;++c)b=31*b+a.charCodeAt(c)>>>0;return b}
;var kc=RegExp("^(?:([^:/?#.]+):)?(?://(?:([^\\\\/?#]*)@)?([^\\\\/?#]*?)(?::([0-9]+))?(?=[\\\\/?#]|$))?([^?#]+)?(?:\\?([^#]*))?(?:#([\\s\\S]*))?$");function lc(a){return a?decodeURI(a):a}
function mc(a,b){return b.match(kc)[a]||null}
function nc(a){return lc(mc(3,a))}
function oc(a){var b=a.match(kc);a=b[5];var c=b[6];b=b[7];var d="";a&&(d+=a);c&&(d+="?"+c);b&&(d+="#"+b);return d}
function pc(a,b){if(!b)return a;var c=a.indexOf("#");0>c&&(c=a.length);var d=a.indexOf("?");if(0>d||d>c){d=c;var e=""}else e=a.substring(d+1,c);a=[a.slice(0,d),e,a.slice(c)];c=a[1];a[1]=b?c?c+"&"+b:b:c;return a[0]+(a[1]?"?"+a[1]:"")+a[2]}
function qc(a,b,c){if(Array.isArray(b))for(var d=0;d<b.length;d++)qc(a,String(b[d]),c);else null!=b&&c.push(a+(""===b?"":"="+encodeURIComponent(String(b))))}
function rc(a,b){var c=[];for(b=b||0;b<a.length;b+=2)qc(a[b],a[b+1],c);return c.join("&")}
function sc(a){var b=[],c;for(c in a)qc(c,a[c],b);return b.join("&")}
function tc(a,b){var c=2==arguments.length?rc(arguments[1],0):rc(arguments,1);return pc(a,c)}
function uc(a,b){b=sc(b);return pc(a,b)}
function vc(a,b,c){c=null!=c?"="+encodeURIComponent(String(c)):"";return pc(a,b+c)}
function wc(a,b,c,d){for(var e=c.length;0<=(b=a.indexOf(c,b))&&b<d;){var f=a.charCodeAt(b-1);if(38==f||63==f)if(f=a.charCodeAt(b+e),!f||61==f||38==f||35==f)return b;b+=e+1}return-1}
var xc=/#|$/,yc=/[?&]($|#)/;function Cc(a,b){for(var c=a.search(xc),d=0,e,f=[];0<=(e=wc(a,d,b,c));)f.push(a.substring(d,e)),d=Math.min(a.indexOf("&",e)+1||c,c);f.push(a.slice(d));return f.join("").replace(yc,"$1")}
;function Dc(a){z.setTimeout(function(){throw a;},0)}
;function Ec(){return C("iPhone")&&!C("iPod")&&!C("iPad")}
function Fc(){var a=Sb(),b="";C("Windows")?(b=/Windows (?:NT|Phone) ([0-9.]+)/,b=(a=b.exec(a))?a[1]:"0.0"):Ec()||C("iPad")||C("iPod")?(b=/(?:iPhone|iPod|iPad|CPU)\s+OS\s+(\S+)/,b=(a=b.exec(a))&&a[1].replace(/_/g,".")):C("Macintosh")?(b=/Mac OS X ([0-9_.]+)/,b=(a=b.exec(a))?a[1].replace(/_/g,"."):"10"):-1!=Sb().toLowerCase().indexOf("kaios")?(b=/(?:KaiOS)\/(\S+)/i,b=(a=b.exec(a))&&a[1]):C("Android")?(b=/Android\s+([^\);]+)(\)|;)/,b=(a=b.exec(a))&&a[1]):C("CrOS")&&(b=/(?:CrOS\s+(?:i686|x86_64)\s+([0-9.]+))/,
b=(a=b.exec(a))&&a[1]);a=0;b=Jb(String(b||"")).split(".");for(var c=Jb("12").split("."),d=Math.max(b.length,c.length),e=0;0==a&&e<d;e++){var f=b[e]||"",g=c[e]||"";do{f=/(\d*)(\D*)(.*)/.exec(f)||["","","",""];g=/(\d*)(\D*)(.*)/.exec(g)||["","","",""];if(0==f[0].length&&0==g[0].length)break;a=Kb(0==f[1].length?0:parseInt(f[1],10),0==g[1].length?0:parseInt(g[1],10))||Kb(0==f[2].length,0==g[2].length)||Kb(f[2],g[2]);f=f[3];g=g[3]}while(0==a)}}
;function Gc(a){Gc[" "](a);return a}
Gc[" "]=function(){};var Hc=C("Opera"),Ic=Tb(),Jc=C("Edge"),Kc=C("Gecko")&&!(-1!=Sb().toLowerCase().indexOf("webkit")&&!C("Edge"))&&!(C("Trident")||C("MSIE"))&&!C("Edge"),Lc=-1!=Sb().toLowerCase().indexOf("webkit")&&!C("Edge"),Mc=C("Android");function Nc(){var a=z.document;return a?a.documentMode:void 0}
var Oc;a:{var Pc="",Qc=function(){var a=Sb();if(Kc)return/rv:([^\);]+)(\)|;)/.exec(a);if(Jc)return/Edge\/([\d\.]+)/.exec(a);if(Ic)return/\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/.exec(a);if(Lc)return/WebKit\/(\S+)/.exec(a);if(Hc)return/(?:Version)[ \/]?(\S+)/.exec(a)}();
Qc&&(Pc=Qc?Qc[1]:"");if(Ic){var Rc=Nc();if(null!=Rc&&Rc>parseFloat(Pc)){Oc=String(Rc);break a}}Oc=Pc}var Sc=Oc,Tc;if(z.document&&Ic){var Uc=Nc();Tc=Uc?Uc:parseInt(Sc,10)||void 0}else Tc=void 0;var Vc=Tc;var Wc=Ec()||C("iPod"),Xc=C("iPad");Xb();Wb();var Yc=Vb()&&!(Ec()||C("iPad")||C("iPod"));var Zc={},$c=null;function ad(a,b){Pa(a);void 0===b&&(b=0);bd();b=Zc[b];for(var c=Array(Math.floor(a.length/3)),d=b[64]||"",e=0,f=0;e<a.length-2;e+=3){var g=a[e],h=a[e+1],k=a[e+2],m=b[g>>2];g=b[(g&3)<<4|h>>4];h=b[(h&15)<<2|k>>6];k=b[k&63];c[f++]=""+m+g+h+k}m=0;k=d;switch(a.length-e){case 2:m=a[e+1],k=b[(m&15)<<2]||d;case 1:a=a[e],c[f]=""+b[a>>2]+b[(a&3)<<4|m>>4]+k+d}return c.join("")}
function cd(a){var b=a.length,c=3*b/4;c%3?c=Math.floor(c):-1!="=.".indexOf(a[b-1])&&(c=-1!="=.".indexOf(a[b-2])?c-2:c-1);var d=new Uint8Array(c),e=0;dd(a,function(f){d[e++]=f});
return e!==c?d.subarray(0,e):d}
function dd(a,b){function c(k){for(;d<a.length;){var m=a.charAt(d++),q=$c[m];if(null!=q)return q;if(!/^[\s\xa0]*$/.test(m))throw Error("Unknown base64 encoding at char: "+m);}return k}
bd();for(var d=0;;){var e=c(-1),f=c(0),g=c(64),h=c(64);if(64===h&&-1===e)break;b(e<<2|f>>4);64!=g&&(b(f<<4&240|g>>2),64!=h&&b(g<<6&192|h))}}
function bd(){if(!$c){$c={};for(var a="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split(""),b=["+/=","+/","-_=","-_.","-_"],c=0;5>c;c++){var d=a.concat(b[c].split(""));Zc[c]=d;for(var e=0;e<d.length;e++){var f=d[e];void 0===$c[f]&&($c[f]=e)}}}}
;var ed="undefined"!==typeof Uint8Array;function fd(a){return ed&&null!=a&&a instanceof Uint8Array}
var gd={};var hd;function id(a){if(a!==gd)throw Error("illegal external caller");}
function jd(a,b){id(b);this.ba=a;if(null!=a&&0===a.length)throw Error("ByteString should be constructed with non-empty values");}
function kd(){return hd||(hd=new jd(null,gd))}
jd.prototype.Oa=function(){return null==this.ba};
jd.prototype.sizeBytes=function(){id(gd);var a=this.ba;null==a||fd(a)||("string"===typeof a?a=cd(a):(Oa(a),a=null));return(a=null==a?a:this.ba=a)?a.length:0};var ld="function"===typeof Symbol&&"symbol"===typeof Symbol()?Symbol():void 0;function md(a,b){if(ld)return a[ld]|=b;if(void 0!==a.fa)return a.fa|=b;Object.defineProperties(a,{fa:{value:b,configurable:!0,writable:!0,enumerable:!1}});return b}
function nd(a,b){ld?a[ld]&&(a[ld]&=~b):void 0!==a.fa&&(a.fa&=~b)}
function od(a){var b;ld?b=a[ld]:b=a.fa;return null==b?0:b}
function pd(a,b){ld?a[ld]=b:void 0!==a.fa?a.fa=b:Object.defineProperties(a,{fa:{value:b,configurable:!0,writable:!0,enumerable:!1}})}
function qd(a){md(a,1);return a}
function rd(a){return!!(od(a)&2)}
function wd(a){md(a,16);return a}
function xd(a,b){pd(b,(a|0)&-51)}
function yd(a,b){pd(b,(a|18)&-41)}
function zd(a,b){var c=od(a);(c&b)!==b&&(Object.isFrozen(a)&&(a=Array.prototype.slice.call(a)),pd(a,c|b));return a}
;var Ad={};function Bd(a){return null!==a&&"object"===typeof a&&!Array.isArray(a)&&a.constructor===Object}
var Cd,Dd,Ed=[];pd(Ed,23);Dd=Object.freeze(Ed);function Fd(a){if(rd(a.C))throw Error("Cannot mutate an immutable Message");}
function Gd(a){var b=a.length;(b=b?a[b-1]:void 0)&&Bd(b)?b.g=1:(b={},a.push((b.g=1,b)))}
;function Hd(a){return a.displayName||a.name||"unknown type name"}
function Id(a,b){if(!(a instanceof b))throw Error("Expected instanceof "+Hd(b)+" but got "+(a&&Hd(a.constructor)));return a}
;var Jd;function Kd(a){switch(typeof a){case "number":return isFinite(a)?a:String(a);case "object":if(a)if(Array.isArray(a)){if(0!==(od(a)&128))return a=Array.prototype.slice.call(a),Gd(a),a}else{if(fd(a))return ad(a);if(a instanceof jd){var b=a.ba;return null==b?"":"string"===typeof b?b:a.ba=ad(b)}}}return a}
;function Ld(a,b,c,d){if(null!=a){if(Array.isArray(a))a=Md(a,b,c,void 0!==d);else if(Bd(a)){var e={},f;for(f in a)e[f]=Ld(a[f],b,c,d);a=e}else a=b(a,d);return a}}
function Md(a,b,c,d){var e=od(a);d=d?!!(e&16):void 0;a=Array.prototype.slice.call(a);for(var f=0;f<a.length;f++)a[f]=Ld(a[f],b,c,d);c(e,a);return a}
function Nd(a){return a.jb===Ad?a.toJSON():Kd(a)}
function Od(a){if(!a)return a;if("object"===typeof a){if(fd(a))return new Uint8Array(a);if(a.jb===Ad)return a.clone()}return a}
function Pd(a,b){a&128&&Gd(b)}
;function Qd(a){return a.j||(a.j=a.C[a.l+a.ta]={})}
function Rd(a,b,c){return-1===b?null:b>=a.l?a.j?a.j[b]:void 0:c&&a.j&&(c=a.j[b],null!=c)?c:a.C[b+a.ta]}
function D(a,b,c,d){Fd(a);return Sd(a,b,c,d)}
function Sd(a,b,c,d){a.m&&(a.m=void 0);if(b>=a.l||d)return Qd(a)[b]=c,a;a.C[b+a.ta]=c;(c=a.j)&&b in c&&delete c[b];return a}
function Td(a,b){a&&rd(b.C)&&rd(a.C);return a}
function Ud(a,b,c,d,e){var f=Rd(a,b,d);Array.isArray(f)||(f=Dd);var g=od(f);g&1||qd(f);if(e)g&2||md(f,2),c&1||Object.freeze(f);else{e=!(c&2);var h=g&2;c&1||!h?e&&g&16&&!h&&nd(f,16):(f=qd(Array.prototype.slice.call(f)),Sd(a,b,f,d))}return f}
function Vd(a,b,c,d){Fd(a);(c=Wd(a,c))&&c!==b&&null!=d&&Sd(a,c,void 0,!1);return Sd(a,b,d)}
function Wd(a,b){for(var c=0,d=0;d<b.length;d++){var e=b[d];null!=Rd(a,e)&&(0!==c&&Sd(a,c,void 0,!1),c=e)}return c}
function Xd(a,b,c,d){var e=d=void 0===d?!1:d,f=Rd(a,c,e);var g=!1;var h=null==f||"object"!==typeof f||(g=Array.isArray(f))||f.jb!==Ad?g?new b(f):void 0:f;h!==f&&null!=h&&(Sd(a,c,h,e),md(h.C,od(a.C)&-33));b=Td(h,a);if(null==b)return b;rd(a.C)||(e=Yd(b),e!==b&&(b=e,Sd(a,c,b,d)));return Td(b,a)}
function Zd(a,b,c,d,e,f){a.h||(a.h={});var g=a.h[c],h=Ud(a,c,3,d,f);if(g)f||(Object.isFrozen(g)?e||(g=Array.prototype.slice.call(g),a.h[c]=g):e&&Object.freeze(g));else{g=[];var k=!!(od(a.C)&16),m=rd(h);!f&&m&&(h=qd(Array.prototype.slice.call(h)),Sd(a,c,h,d));d=m;for(var q=0;q<h.length;q++){var r=h[q];var w=b;var t=k,y=!1;y=void 0===y?!1:y;t=void 0===t?!1:t;w=Array.isArray(r)?new w(t?wd(r):r):y?new w:void 0;void 0!==w&&(d=d||rd(r),g.push(w),m&&md(w.C,2))}a.h[c]=g;a=h;b=!d;Object.isFrozen(a)||(c=od(a)|
33,pd(a,b?c|8:c&-9));(f||e&&m)&&md(g,2);(f||e)&&Object.freeze(g)}return g}
function $d(a,b,c,d){var e=rd(a.C);b=Zd(a,b,c,d,e,e);a=Ud(a,c,3,d,e);if(!e&&a&&!(od(a)&8)){for(e=0;e<b.length;e++)c=b[e],d=Yd(c),c!==d&&(b[e]=d,a[e]=b[e].C);md(a,8)}return b}
function G(a,b,c,d){Fd(a);null!=d?Id(d,b):d=void 0;return Sd(a,c,d)}
function ae(a,b,c,d,e){Fd(a);null!=e?Id(e,b):e=void 0;Vd(a,c,d,e)}
function be(a,b,c,d,e){Fd(a);if(null!=d){var f=qd([]);for(var g=!1,h=0;h<d.length;h++)f[h]=Id(d[h],b).C,g=g||rd(f[h]);a.h||(a.h={});a.h[c]=d;b=f;g?nd(b,8):md(b,8)}else a.h&&(a.h[c]=void 0),f=Dd;return Sd(a,c,f,e)}
function ce(a,b,c,d){Fd(a);var e=Zd(a,c,b,void 0,!1,!1);c=null!=d?Id(d,c):new c;a=Ud(a,b,2,void 0,!1);e.push(c);a.push(c.C);rd(c.C)&&nd(a,8)}
function de(a,b){return Rd(a,b)}
function ee(a,b){return null==a?b:a}
;function fe(a,b,c){c=void 0===c?yd:c;if(null!=a){if(ed&&a instanceof Uint8Array)return a.length?new jd(new Uint8Array(a),gd):kd();if(Array.isArray(a)){var d=od(a);if(d&2)return a;if(b&&!(d&32)&&(d&16||0===d))return pd(a,d|2),a;a=Md(a,fe,c,!0);b=od(a);b&4&&b&2&&Object.freeze(a);return a}return a.jb===Ad?ge(a):a}}
function he(a,b,c,d,e,f,g){(a=a.h&&a.h[c])?(d=0<a.length?a[0].constructor:void 0,f=od(a),f&2||(a=hb(a,ge),yd(f,a),Object.freeze(a)),be(b,d,c,a,e)):D(b,c,fe(d,f,g),e)}
function ge(a){if(rd(a.C))return a;a=ie(a,!0);md(a.C,2);return a}
function ie(a,b){var c=a.C,d=wd([]),e=a.constructor.h;e&&d.push(e);0!==(od(c)&128)&&Gd(d);b=b||rd(a.C)?yd:xd;e=a.constructor;od(d);Jd=d;d=new e(d);Jd=void 0;a.Na&&(d.Na=a.Na.slice());e=!!(od(c)&16);for(var f=0;f<c.length;f++){var g=c[f];if(f===c.length-1&&Bd(g))for(var h in g){var k=+h;Number.isNaN(k)?Qd(d)[k]=g[k]:he(a,d,k,g[h],!0,e,b)}else he(a,d,f-a.ta,g,!1,e,b)}return d}
function Yd(a){if(!rd(a.C))return a;var b=ie(a,!1);b.m=a;return b}
;function H(a,b,c){null==a&&(a=Jd);Jd=void 0;var d=this.constructor.i||0,e=0<d,f=this.constructor.h,g=!1;if(null==a){a=f?[f]:[];var h=!0;pd(a,48)}else{if(!Array.isArray(a))throw Error();if(f&&f!==a[0])throw Error();var k=md(a,0),m=k;if(h=0!==(16&m))(g=0!==(32&m))||(m|=32);if(e)if(128&m)d=0;else{if(0<a.length){var q=a[a.length-1];if(Bd(q)&&"g"in q){d=0;m|=128;delete q.g;var r=!0,w;for(w in q){r=!1;break}r&&a.pop()}}}else if(128&m)throw Error();k!==m&&pd(a,m)}this.ta=(f?0:-1)-d;this.h=void 0;this.C=
a;a:{f=this.C.length;d=f-1;if(f&&(f=this.C[d],Bd(f))){this.j=f;this.l=d-this.ta;break a}void 0!==b&&-1<b?(this.l=Math.max(b,d+1-this.ta),this.j=void 0):this.l=Number.MAX_VALUE}if(!e&&this.j&&"g"in this.j)throw Error('Unexpected "g" flag in sparse object of message that is not a group type.');if(c){b=h&&!g&&!0;e=this.l;var t;for(h=0;h<c.length;h++)g=c[h],g<e?(g+=this.ta,(d=a[g])?je(d,b):a[g]=Dd):(t||(t=Qd(this)),(d=t[g])?je(d,b):t[g]=Dd)}}
H.prototype.toJSON=function(){var a=this.C,b;Cd?b=a:b=Md(a,Nd,Pd);return b};
function ke(a){Cd=!0;try{return JSON.stringify(a.toJSON(),le)}finally{Cd=!1}}
H.prototype.clone=function(){var a=Md(this.C,Od,xd);wd(a);Jd=a;a=new this.constructor(a);Jd=void 0;me(a,this);return a};
function je(a,b){if(Array.isArray(a)){var c=od(a),d=1;!b||c&2||(d|=16);(c&d)!==d&&pd(a,c|d)}}
H.prototype.jb=Ad;H.prototype.toString=function(){return this.C.toString()};
function le(a,b){return Kd(b)}
function me(a,b){b.Na&&(a.Na=b.Na.slice());var c=b.h;if(c){b=b.j;for(var d in c){var e=c[d];if(e){var f=!(!b||!b[d]),g=+d;if(Array.isArray(e)){if(e.length)for(f=$d(a,e[0].constructor,g,f),g=0;g<Math.min(f.length,e.length);g++)me(f[g],e[g])}else throw Error("unexpected object: type: "+Oa(e)+": "+e);}}}}
;function ne(a){var b=this.h,c=this.i;return this.isRepeated?$d(a,b,c,!0):Xd(a,b,c,!0)}
;function oe(a){this.Sb=a}
;function pe(a,b,c){this.i=a;this.l=b;this.h=c||[];this.Aa=new Map}
l=pe.prototype;l.Fc=function(a){var b=Ja.apply(1,arguments),c=this.ub(b);c?c.push(new oe(a)):this.sc(a,b)};
l.sc=function(a){this.Aa.set(this.Zb(Ja.apply(1,arguments)),[new oe(a)])};
l.ub=function(){var a=this.Zb(Ja.apply(0,arguments));return this.Aa.has(a)?this.Aa.get(a):void 0};
l.Rc=function(){var a=this.ub(Ja.apply(0,arguments));return a&&a.length?a[0]:void 0};
l.clear=function(){this.Aa.clear()};
l.Zb=function(){var a=Ja.apply(0,arguments);return a?a.join(","):"key"};function qe(a,b){pe.call(this,a,3,b)}
u(qe,pe);qe.prototype.j=function(a){var b=Ja.apply(1,arguments),c=0,d=this.Rc(b);d&&(c=d.Sb);this.sc(c+a,b)};function re(a,b){pe.call(this,a,2,b)}
u(re,pe);re.prototype.j=function(a){this.Fc(a,Ja.apply(1,arguments))};function se(a){a&&"function"==typeof a.dispose&&a.dispose()}
;function te(a){for(var b=0,c=arguments.length;b<c;++b){var d=arguments[b];Pa(d)?te.apply(null,d):se(d)}}
;function J(){this.M=this.M;this.v=this.v}
J.prototype.M=!1;J.prototype.h=function(){return this.M};
J.prototype.dispose=function(){this.M||(this.M=!0,this.B())};
function ue(a,b){ve(a,Ya(se,b))}
function ve(a,b){a.M?b():(a.v||(a.v=[]),a.v.push(b))}
J.prototype.B=function(){if(this.v)for(;this.v.length;)this.v.shift()()};function we(a,b){this.type=a;this.h=this.target=b;this.defaultPrevented=this.j=!1}
we.prototype.stopPropagation=function(){this.j=!0};
we.prototype.preventDefault=function(){this.defaultPrevented=!0};function xe(a){var b=B("window.location.href");null==a&&(a='Unknown Error of type "null/undefined"');if("string"===typeof a)return{message:a,name:"Unknown error",lineNumber:"Not available",fileName:b,stack:"Not available"};var c=!1;try{var d=a.lineNumber||a.line||"Not available"}catch(g){d="Not available",c=!0}try{var e=a.fileName||a.filename||a.sourceURL||z.$googDebugFname||b}catch(g){e="Not available",c=!0}b=ye(a);if(!(!c&&a.lineNumber&&a.fileName&&a.stack&&a.message&&a.name)){c=a.message;if(null==
c){if(a.constructor&&a.constructor instanceof Function){if(a.constructor.name)c=a.constructor.name;else if(c=a.constructor,ze[c])c=ze[c];else{c=String(c);if(!ze[c]){var f=/function\s+([^\(]+)/m.exec(c);ze[c]=f?f[1]:"[Anonymous]"}c=ze[c]}c='Unknown Error of type "'+c+'"'}else c="Unknown Error of unknown type";"function"===typeof a.toString&&Object.prototype.toString!==a.toString&&(c+=": "+a.toString())}return{message:c,name:a.name||"UnknownError",lineNumber:d,fileName:e,stack:b||"Not available"}}a.stack=
b;return{message:a.message,name:a.name,lineNumber:a.lineNumber,fileName:a.fileName,stack:a.stack}}
function ye(a,b){b||(b={});b[Ae(a)]=!0;var c=a.stack||"";(a=a.cause)&&!b[Ae(a)]&&(c+="\nCaused by: ",a.stack&&0==a.stack.indexOf(a.toString())||(c+="string"===typeof a?a:a.message+"\n"),c+=ye(a,b));return c}
function Ae(a){var b="";"function"===typeof a.toString&&(b=""+a);return b+a.stack}
var ze={};var Be=function(){if(!z.addEventListener||!Object.defineProperty)return!1;var a=!1,b=Object.defineProperty({},"passive",{get:function(){a=!0}});
try{z.addEventListener("test",function(){},b),z.removeEventListener("test",function(){},b)}catch(c){}return a}();function Ce(a,b){we.call(this,a?a.type:"");this.relatedTarget=this.h=this.target=null;this.button=this.screenY=this.screenX=this.clientY=this.clientX=0;this.key="";this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.altKey=this.ctrlKey=!1;this.state=null;this.pointerId=0;this.pointerType="";this.i=null;a&&this.init(a,b)}
Za(Ce,we);var De={2:"touch",3:"pen",4:"mouse"};
Ce.prototype.init=function(a,b){var c=this.type=a.type,d=a.changedTouches&&a.changedTouches.length?a.changedTouches[0]:null;this.target=a.target||a.srcElement;this.h=b;if(b=a.relatedTarget){if(Kc){a:{try{Gc(b.nodeName);var e=!0;break a}catch(f){}e=!1}e||(b=null)}}else"mouseover"==c?b=a.fromElement:"mouseout"==c&&(b=a.toElement);this.relatedTarget=b;d?(this.clientX=void 0!==d.clientX?d.clientX:d.pageX,this.clientY=void 0!==d.clientY?d.clientY:d.pageY,this.screenX=d.screenX||0,this.screenY=d.screenY||
0):(this.clientX=void 0!==a.clientX?a.clientX:a.pageX,this.clientY=void 0!==a.clientY?a.clientY:a.pageY,this.screenX=a.screenX||0,this.screenY=a.screenY||0);this.button=a.button;this.keyCode=a.keyCode||0;this.key=a.key||"";this.charCode=a.charCode||("keypress"==c?a.keyCode:0);this.ctrlKey=a.ctrlKey;this.altKey=a.altKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.pointerId=a.pointerId||0;this.pointerType="string"===typeof a.pointerType?a.pointerType:De[a.pointerType]||"";this.state=a.state;
this.i=a;a.defaultPrevented&&Ce.X.preventDefault.call(this)};
Ce.prototype.stopPropagation=function(){Ce.X.stopPropagation.call(this);this.i.stopPropagation?this.i.stopPropagation():this.i.cancelBubble=!0};
Ce.prototype.preventDefault=function(){Ce.X.preventDefault.call(this);var a=this.i;a.preventDefault?a.preventDefault():a.returnValue=!1};var Je="closure_listenable_"+(1E6*Math.random()|0);var Ke=0;function Le(a,b,c,d,e){this.listener=a;this.proxy=null;this.src=b;this.type=c;this.capture=!!d;this.hb=e;this.key=++Ke;this.Pa=this.cb=!1}
function Me(a){a.Pa=!0;a.listener=null;a.proxy=null;a.src=null;a.hb=null}
;function Ne(a){this.src=a;this.listeners={};this.h=0}
Ne.prototype.add=function(a,b,c,d,e){var f=a.toString();a=this.listeners[f];a||(a=this.listeners[f]=[],this.h++);var g=Oe(a,b,d,e);-1<g?(b=a[g],c||(b.cb=!1)):(b=new Le(b,this.src,f,!!d,e),b.cb=c,a.push(b));return b};
Ne.prototype.remove=function(a,b,c,d){a=a.toString();if(!(a in this.listeners))return!1;var e=this.listeners[a];b=Oe(e,b,c,d);return-1<b?(Me(e[b]),Array.prototype.splice.call(e,b,1),0==e.length&&(delete this.listeners[a],this.h--),!0):!1};
function Pe(a,b){var c=b.type;c in a.listeners&&kb(a.listeners[c],b)&&(Me(b),0==a.listeners[c].length&&(delete a.listeners[c],a.h--))}
function Oe(a,b,c,d){for(var e=0;e<a.length;++e){var f=a[e];if(!f.Pa&&f.listener==b&&f.capture==!!c&&f.hb==d)return e}return-1}
;var Qe="closure_lm_"+(1E6*Math.random()|0),Re={},Se=0;function Te(a,b,c,d,e){if(d&&d.once)Ue(a,b,c,d,e);else if(Array.isArray(b))for(var f=0;f<b.length;f++)Te(a,b[f],c,d,e);else c=Ve(c),a&&a[Je]?a.ha(b,c,Qa(d)?!!d.capture:!!d,e):We(a,b,c,!1,d,e)}
function We(a,b,c,d,e,f){if(!b)throw Error("Invalid event type");var g=Qa(e)?!!e.capture:!!e,h=Xe(a);h||(a[Qe]=h=new Ne(a));c=h.add(b,c,d,g,f);if(!c.proxy){d=Ye();c.proxy=d;d.src=a;d.listener=c;if(a.addEventListener)Be||(e=g),void 0===e&&(e=!1),a.addEventListener(b.toString(),d,e);else if(a.attachEvent)a.attachEvent(Ze(b.toString()),d);else if(a.addListener&&a.removeListener)a.addListener(d);else throw Error("addEventListener and attachEvent are unavailable.");Se++}}
function Ye(){function a(c){return b.call(a.src,a.listener,c)}
var b=$e;return a}
function Ue(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)Ue(a,b[f],c,d,e);else c=Ve(c),a&&a[Je]?a.l.add(String(b),c,!0,Qa(d)?!!d.capture:!!d,e):We(a,b,c,!0,d,e)}
function af(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)af(a,b[f],c,d,e);else(d=Qa(d)?!!d.capture:!!d,c=Ve(c),a&&a[Je])?a.l.remove(String(b),c,d,e):a&&(a=Xe(a))&&(b=a.listeners[b.toString()],a=-1,b&&(a=Oe(b,c,d,e)),(c=-1<a?b[a]:null)&&bf(c))}
function bf(a){if("number"!==typeof a&&a&&!a.Pa){var b=a.src;if(b&&b[Je])Pe(b.l,a);else{var c=a.type,d=a.proxy;b.removeEventListener?b.removeEventListener(c,d,a.capture):b.detachEvent?b.detachEvent(Ze(c),d):b.addListener&&b.removeListener&&b.removeListener(d);Se--;(c=Xe(b))?(Pe(c,a),0==c.h&&(c.src=null,b[Qe]=null)):Me(a)}}}
function Ze(a){return a in Re?Re[a]:Re[a]="on"+a}
function $e(a,b){if(a.Pa)a=!0;else{b=new Ce(b,this);var c=a.listener,d=a.hb||a.src;a.cb&&bf(a);a=c.call(d,b)}return a}
function Xe(a){a=a[Qe];return a instanceof Ne?a:null}
var cf="__closure_events_fn_"+(1E9*Math.random()>>>0);function Ve(a){if("function"===typeof a)return a;a[cf]||(a[cf]=function(b){return a.handleEvent(b)});
return a[cf]}
;function df(){J.call(this);this.l=new Ne(this);this.Cc=this;this.ja=null}
Za(df,J);df.prototype[Je]=!0;df.prototype.addEventListener=function(a,b,c,d){Te(this,a,b,c,d)};
df.prototype.removeEventListener=function(a,b,c,d){af(this,a,b,c,d)};
function ef(a,b){var c=a.ja;if(c){var d=[];for(var e=1;c;c=c.ja)d.push(c),++e}a=a.Cc;c=b.type||b;"string"===typeof b?b=new we(b,a):b instanceof we?b.target=b.target||a:(e=b,b=new we(c,a),xb(b,e));e=!0;if(d)for(var f=d.length-1;!b.j&&0<=f;f--){var g=b.h=d[f];e=ff(g,c,!0,b)&&e}b.j||(g=b.h=a,e=ff(g,c,!0,b)&&e,b.j||(e=ff(g,c,!1,b)&&e));if(d)for(f=0;!b.j&&f<d.length;f++)g=b.h=d[f],e=ff(g,c,!1,b)&&e}
df.prototype.B=function(){df.X.B.call(this);if(this.l){var a=this.l,b=0,c;for(c in a.listeners){for(var d=a.listeners[c],e=0;e<d.length;e++)++b,Me(d[e]);delete a.listeners[c];a.h--}}this.ja=null};
df.prototype.ha=function(a,b,c,d){return this.l.add(String(a),b,!1,c,d)};
function ff(a,b,c,d){b=a.l.listeners[String(b)];if(!b)return!0;b=b.concat();for(var e=!0,f=0;f<b.length;++f){var g=b[f];if(g&&!g.Pa&&g.capture==c){var h=g.listener,k=g.hb||g.src;g.cb&&Pe(a.l,g);e=!1!==h.call(k,d)&&e}}return e&&!d.defaultPrevented}
;function gf(a,b){this.j=a;this.l=b;this.i=0;this.h=null}
gf.prototype.get=function(){if(0<this.i){this.i--;var a=this.h;this.h=a.next;a.next=null}else a=this.j();return a};
function hf(a,b){a.l(b);100>a.i&&(a.i++,b.next=a.h,a.h=b)}
;function jf(a,b){return a+Math.random()*(b-a)}
;function kf(a,b){this.x=void 0!==a?a:0;this.y=void 0!==b?b:0}
l=kf.prototype;l.clone=function(){return new kf(this.x,this.y)};
l.equals=function(a){return a instanceof kf&&(this==a?!0:this&&a?this.x==a.x&&this.y==a.y:!1)};
l.ceil=function(){this.x=Math.ceil(this.x);this.y=Math.ceil(this.y);return this};
l.floor=function(){this.x=Math.floor(this.x);this.y=Math.floor(this.y);return this};
l.round=function(){this.x=Math.round(this.x);this.y=Math.round(this.y);return this};
l.scale=function(a,b){this.x*=a;this.y*="number"===typeof b?b:a;return this};function lf(a,b){this.width=a;this.height=b}
l=lf.prototype;l.clone=function(){return new lf(this.width,this.height)};
l.aspectRatio=function(){return this.width/this.height};
l.Oa=function(){return!(this.width*this.height)};
l.ceil=function(){this.width=Math.ceil(this.width);this.height=Math.ceil(this.height);return this};
l.floor=function(){this.width=Math.floor(this.width);this.height=Math.floor(this.height);return this};
l.round=function(){this.width=Math.round(this.width);this.height=Math.round(this.height);return this};
l.scale=function(a,b){this.width*=a;this.height*="number"===typeof b?b:a;return this};function mf(a){var b=document;return"string"===typeof a?b.getElementById(a):a}
function nf(a){var b=document;a=String(a);"application/xhtml+xml"===b.contentType&&(a=a.toLowerCase());return b.createElement(a)}
function of(a,b){for(var c=0;a;){if(b(a))return a;a=a.parentNode;c++}return null}
;var pf;function qf(){var a=z.MessageChannel;"undefined"===typeof a&&"undefined"!==typeof window&&window.postMessage&&window.addEventListener&&!C("Presto")&&(a=function(){var e=nf("IFRAME");e.style.display="none";document.documentElement.appendChild(e);var f=e.contentWindow;e=f.document;e.open();e.close();var g="callImmediate"+Math.random(),h="file:"==f.location.protocol?"*":f.location.protocol+"//"+f.location.host;e=Xa(function(k){if(("*"==h||k.origin==h)&&k.data==g)this.port1.onmessage()},this);
f.addEventListener("message",e,!1);this.port1={};this.port2={postMessage:function(){f.postMessage(g,h)}}});
if("undefined"!==typeof a&&!Tb()){var b=new a,c={},d=c;b.port1.onmessage=function(){if(void 0!==c.next){c=c.next;var e=c.Rb;c.Rb=null;e()}};
return function(e){d.next={Rb:e};d=d.next;b.port2.postMessage(0)}}return function(e){z.setTimeout(e,0)}}
;function rf(){this.i=this.h=null}
rf.prototype.add=function(a,b){var c=sf.get();c.set(a,b);this.i?this.i.next=c:this.h=c;this.i=c};
rf.prototype.remove=function(){var a=null;this.h&&(a=this.h,this.h=this.h.next,this.h||(this.i=null),a.next=null);return a};
var sf=new gf(function(){return new tf},function(a){return a.reset()});
function tf(){this.next=this.scope=this.h=null}
tf.prototype.set=function(a,b){this.h=a;this.scope=b;this.next=null};
tf.prototype.reset=function(){this.next=this.scope=this.h=null};var uf,vf=!1,wf=new rf;function xf(a,b){uf||yf();vf||(uf(),vf=!0);wf.add(a,b)}
function yf(){if(z.Promise&&z.Promise.resolve){var a=z.Promise.resolve(void 0);uf=function(){a.then(zf)}}else uf=function(){var b=zf;
"function"!==typeof z.setImmediate||z.Window&&z.Window.prototype&&!C("Edge")&&z.Window.prototype.setImmediate==z.setImmediate?(pf||(pf=qf()),pf(b)):z.setImmediate(b)}}
function zf(){for(var a;a=wf.remove();){try{a.h.call(a.scope)}catch(b){Dc(b)}hf(sf,a)}vf=!1}
;function Af(a){this.h=0;this.v=void 0;this.l=this.i=this.j=null;this.m=this.o=!1;if(a!=cb)try{var b=this;a.call(void 0,function(c){Bf(b,2,c)},function(c){Bf(b,3,c)})}catch(c){Bf(this,3,c)}}
function Cf(){this.next=this.context=this.i=this.j=this.h=null;this.l=!1}
Cf.prototype.reset=function(){this.context=this.i=this.j=this.h=null;this.l=!1};
var Df=new gf(function(){return new Cf},function(a){a.reset()});
function Ef(a,b,c){var d=Df.get();d.j=a;d.i=b;d.context=c;return d}
function Ff(a){return new Af(function(b,c){c(a)})}
Af.prototype.then=function(a,b,c){return Gf(this,"function"===typeof a?a:null,"function"===typeof b?b:null,c)};
Af.prototype.$goog_Thenable=!0;l=Af.prototype;l.qb=function(a,b){return Gf(this,null,a,b)};
l.catch=Af.prototype.qb;l.cancel=function(a){if(0==this.h){var b=new Hf(a);xf(function(){If(this,b)},this)}};
function If(a,b){if(0==a.h)if(a.j){var c=a.j;if(c.i){for(var d=0,e=null,f=null,g=c.i;g&&(g.l||(d++,g.h==a&&(e=g),!(e&&1<d)));g=g.next)e||(f=g);e&&(0==c.h&&1==d?If(c,b):(f?(d=f,d.next==c.l&&(c.l=d),d.next=d.next.next):Jf(c),Kf(c,e,3,b)))}a.j=null}else Bf(a,3,b)}
function Lf(a,b){a.i||2!=a.h&&3!=a.h||Mf(a);a.l?a.l.next=b:a.i=b;a.l=b}
function Gf(a,b,c,d){var e=Ef(null,null,null);e.h=new Af(function(f,g){e.j=b?function(h){try{var k=b.call(d,h);f(k)}catch(m){g(m)}}:f;
e.i=c?function(h){try{var k=c.call(d,h);void 0===k&&h instanceof Hf?g(h):f(k)}catch(m){g(m)}}:g});
e.h.j=a;Lf(a,e);return e.h}
l.Ad=function(a){this.h=0;Bf(this,2,a)};
l.Bd=function(a){this.h=0;Bf(this,3,a)};
function Bf(a,b,c){if(0==a.h){a===c&&(b=3,c=new TypeError("Promise cannot resolve to itself"));a.h=1;a:{var d=c,e=a.Ad,f=a.Bd;if(d instanceof Af){Lf(d,Ef(e||cb,f||null,a));var g=!0}else{if(d)try{var h=!!d.$goog_Thenable}catch(m){h=!1}else h=!1;if(h)d.then(e,f,a),g=!0;else{if(Qa(d))try{var k=d.then;if("function"===typeof k){Nf(d,k,e,f,a);g=!0;break a}}catch(m){f.call(a,m);g=!0;break a}g=!1}}}g||(a.v=c,a.h=b,a.j=null,Mf(a),3!=b||c instanceof Hf||Of(a,c))}}
function Nf(a,b,c,d,e){function f(k){h||(h=!0,d.call(e,k))}
function g(k){h||(h=!0,c.call(e,k))}
var h=!1;try{b.call(a,g,f)}catch(k){f(k)}}
function Mf(a){a.o||(a.o=!0,xf(a.Pc,a))}
function Jf(a){var b=null;a.i&&(b=a.i,a.i=b.next,b.next=null);a.i||(a.l=null);return b}
l.Pc=function(){for(var a;a=Jf(this);)Kf(this,a,this.h,this.v);this.o=!1};
function Kf(a,b,c,d){if(3==c&&b.i&&!b.l)for(;a&&a.m;a=a.j)a.m=!1;if(b.h)b.h.j=null,Pf(b,c,d);else try{b.l?b.j.call(b.context):Pf(b,c,d)}catch(e){Qf.call(null,e)}hf(Df,b)}
function Pf(a,b,c){2==b?a.j.call(a.context,c):a.i&&a.i.call(a.context,c)}
function Of(a,b){a.m=!0;xf(function(){a.m&&Qf.call(null,b)})}
var Qf=Dc;function Hf(a){ab.call(this,a)}
Za(Hf,ab);Hf.prototype.name="cancel";function Rf(a,b){df.call(this);this.j=a||1;this.i=b||z;this.m=Xa(this.yd,this);this.o=Date.now()}
Za(Rf,df);l=Rf.prototype;l.enabled=!1;l.aa=null;function Sf(a,b){a.j=b;a.aa&&a.enabled?(a.stop(),a.start()):a.aa&&a.stop()}
l.yd=function(){if(this.enabled){var a=Date.now()-this.o;0<a&&a<.8*this.j?this.aa=this.i.setTimeout(this.m,this.j-a):(this.aa&&(this.i.clearTimeout(this.aa),this.aa=null),ef(this,"tick"),this.enabled&&(this.stop(),this.start()))}};
l.start=function(){this.enabled=!0;this.aa||(this.aa=this.i.setTimeout(this.m,this.j),this.o=Date.now())};
l.stop=function(){this.enabled=!1;this.aa&&(this.i.clearTimeout(this.aa),this.aa=null)};
l.B=function(){Rf.X.B.call(this);this.stop();delete this.i};
function Tf(a,b,c){if("function"===typeof a)c&&(a=Xa(a,c));else if(a&&"function"==typeof a.handleEvent)a=Xa(a.handleEvent,a);else throw Error("Invalid listener argument");return 2147483647<Number(b)?-1:z.setTimeout(a,b||0)}
;function Uf(a){this.v=a;this.h=new Map;this.o=new Set;this.j=0;this.l=100;this.flushInterval=3E4;this.i=new Rf(this.flushInterval);this.i.ha("tick",this.tb,!1,this);this.m=!1}
l=Uf.prototype;l.pc=function(a){this.m=a;this.l=1};
function Vf(a){a.i.enabled||a.i.start();a.j++;a.j>=a.l&&a.tb()}
l.tb=function(){var a=this.h.values();a=[].concat(ia(a)).filter(function(b){return b.Aa.size});
a.length&&this.v.flush(a,this.m);Wf(a);this.j=0;this.i.enabled&&this.i.stop()};
l.Ob=function(a){var b=Ja.apply(1,arguments);this.h.has(a)||this.h.set(a,new qe(a,b))};
l.Pb=function(a){var b=Ja.apply(1,arguments);this.h.has(a)||this.h.set(a,new re(a,b))};
function Xf(a,b){return a.o.has(b)?void 0:a.h.get(b)}
l.rb=function(a){this.Ac.apply(this,[a,1].concat(ia(Ja.apply(1,arguments))))};
l.Ac=function(a,b){var c=Ja.apply(2,arguments),d=Xf(this,a);d&&d instanceof qe&&(d.j(b,c),Vf(this))};
l.sb=function(a,b){var c=Ja.apply(2,arguments),d=Xf(this,a);d&&d instanceof re&&(d.j(b,c),Vf(this))};
function Wf(a){for(var b=0;b<a.length;b++)a[b].clear()}
;function Yf(a){this.h=a;this.h.Ob("/client_streamz/bg/fiec",{Ma:3,La:"rk"},{Ma:2,La:"ec"})}
function Zf(a){this.h=a;this.h.Pb("/client_streamz/bg/fil",{Ma:3,La:"rk"})}
function $f(a){this.h=a;this.h.Ob("/client_streamz/bg/fsc",{Ma:3,La:"rk"})}
function ag(a){this.h=a;this.h.Pb("/client_streamz/bg/fsl",{Ma:3,La:"rk"})}
;function bg(a){H.call(this,a,-1,cg)}
u(bg,H);function dg(a){H.call(this,a,-1,eg)}
u(dg,H);function fg(a){H.call(this,a)}
u(fg,H);function gg(a){H.call(this,a)}
u(gg,H);var cg=[3,6,4],eg=[1],hg=[1,2,3],ig=[1,2,3];function jg(a){H.call(this,a,-1,kg)}
u(jg,H);var kg=[1];function lg(a){if(!a)return"";if(/^about:(?:blank|srcdoc)$/.test(a))return window.origin||"";a=a.split("#")[0].split("?")[0];a=a.toLowerCase();0==a.indexOf("//")&&(a=window.location.protocol+a);/^[\w\-]*:\/\//.test(a)||(a=window.location.href);var b=a.substring(a.indexOf("://")+3),c=b.indexOf("/");-1!=c&&(b=b.substring(0,c));c=a.substring(0,a.indexOf("://"));if(!c)throw Error("URI is missing protocol: "+a);if("http"!==c&&"https"!==c&&"chrome-extension"!==c&&"moz-extension"!==c&&"file"!==c&&"android-app"!==
c&&"chrome-search"!==c&&"chrome-untrusted"!==c&&"chrome"!==c&&"app"!==c&&"devtools"!==c)throw Error("Invalid URI scheme in origin: "+c);a="";var d=b.indexOf(":");if(-1!=d){var e=b.substring(d+1);b=b.substring(0,d);if("http"===c&&"80"!==e||"https"===c&&"443"!==e)a=":"+e}return c+"://"+b+a}
;function mg(){function a(){e[0]=1732584193;e[1]=4023233417;e[2]=2562383102;e[3]=271733878;e[4]=3285377520;q=m=0}
function b(r){for(var w=g,t=0;64>t;t+=4)w[t/4]=r[t]<<24|r[t+1]<<16|r[t+2]<<8|r[t+3];for(t=16;80>t;t++)r=w[t-3]^w[t-8]^w[t-14]^w[t-16],w[t]=(r<<1|r>>>31)&4294967295;r=e[0];var y=e[1],E=e[2],F=e[3],O=e[4];for(t=0;80>t;t++){if(40>t)if(20>t){var N=F^y&(E^F);var R=1518500249}else N=y^E^F,R=1859775393;else 60>t?(N=y&E|F&(y|E),R=2400959708):(N=y^E^F,R=3395469782);N=((r<<5|r>>>27)&4294967295)+N+O+R+w[t]&4294967295;O=F;F=E;E=(y<<30|y>>>2)&4294967295;y=r;r=N}e[0]=e[0]+r&4294967295;e[1]=e[1]+y&4294967295;e[2]=
e[2]+E&4294967295;e[3]=e[3]+F&4294967295;e[4]=e[4]+O&4294967295}
function c(r,w){if("string"===typeof r){r=unescape(encodeURIComponent(r));for(var t=[],y=0,E=r.length;y<E;++y)t.push(r.charCodeAt(y));r=t}w||(w=r.length);t=0;if(0==m)for(;t+64<w;)b(r.slice(t,t+64)),t+=64,q+=64;for(;t<w;)if(f[m++]=r[t++],q++,64==m)for(m=0,b(f);t+64<w;)b(r.slice(t,t+64)),t+=64,q+=64}
function d(){var r=[],w=8*q;56>m?c(h,56-m):c(h,64-(m-56));for(var t=63;56<=t;t--)f[t]=w&255,w>>>=8;b(f);for(t=w=0;5>t;t++)for(var y=24;0<=y;y-=8)r[w++]=e[t]>>y&255;return r}
for(var e=[],f=[],g=[],h=[128],k=1;64>k;++k)h[k]=0;var m,q;a();return{reset:a,update:c,digest:d,Mc:function(){for(var r=d(),w="",t=0;t<r.length;t++)w+="0123456789ABCDEF".charAt(Math.floor(r[t]/16))+"0123456789ABCDEF".charAt(r[t]%16);return w}}}
;function ng(a,b,c){var d=String(z.location.href);return d&&a&&b?[b,og(lg(d),a,c||null)].join(" "):null}
function og(a,b,c){var d=[],e=[];if(1==(Array.isArray(c)?2:1))return e=[b,a],fb(d,function(h){e.push(h)}),pg(e.join(" "));
var f=[],g=[];fb(c,function(h){g.push(h.key);f.push(h.value)});
c=Math.floor((new Date).getTime()/1E3);e=0==f.length?[c,b,a]:[f.join(":"),c,b,a];fb(d,function(h){e.push(h)});
a=pg(e.join(" "));a=[c,a];0==g.length||a.push(g.join(""));return a.join("_")}
function pg(a){var b=mg();b.update(a);return b.Mc().toLowerCase()}
;var qg={};function rg(a){this.h=a||{cookie:""}}
l=rg.prototype;l.isEnabled=function(){if(!z.navigator.cookieEnabled)return!1;if(!this.Oa())return!0;this.set("TESTCOOKIESENABLED","1",{ib:60});if("1"!==this.get("TESTCOOKIESENABLED"))return!1;this.remove("TESTCOOKIESENABLED");return!0};
l.set=function(a,b,c){var d=!1;if("object"===typeof c){var e=c.Fr;d=c.secure||!1;var f=c.domain||void 0;var g=c.path||void 0;var h=c.ib}if(/[;=\s]/.test(a))throw Error('Invalid cookie name "'+a+'"');if(/[;\r\n]/.test(b))throw Error('Invalid cookie value "'+b+'"');void 0===h&&(h=-1);c=f?";domain="+f:"";g=g?";path="+g:"";d=d?";secure":"";h=0>h?"":0==h?";expires="+(new Date(1970,1,1)).toUTCString():";expires="+(new Date(Date.now()+1E3*h)).toUTCString();this.h.cookie=a+"="+b+c+g+h+d+(null!=e?";samesite="+
e:"")};
l.get=function(a,b){for(var c=a+"=",d=(this.h.cookie||"").split(";"),e=0,f;e<d.length;e++){f=Jb(d[e]);if(0==f.lastIndexOf(c,0))return f.slice(c.length);if(f==a)return""}return b};
l.remove=function(a,b,c){var d=void 0!==this.get(a);this.set(a,"",{ib:0,path:b,domain:c});return d};
l.xb=function(){return sg(this).keys};
l.Oa=function(){return!this.h.cookie};
l.clear=function(){for(var a=sg(this).keys,b=a.length-1;0<=b;b--)this.remove(a[b])};
function sg(a){a=(a.h.cookie||"").split(";");for(var b=[],c=[],d,e,f=0;f<a.length;f++)e=Jb(a[f]),d=e.indexOf("="),-1==d?(b.push(""),c.push(e)):(b.push(e.substring(0,d)),c.push(e.substring(d+1)));return{keys:b,values:c}}
var tg=new rg("undefined"==typeof document?null:document);function ug(a){return!!qg.FPA_SAMESITE_PHASE2_MOD||!(void 0===a||!a)}
function vg(a){a=void 0===a?!1:a;var b=z.__SAPISID||z.__APISID||z.__3PSAPISID||z.__OVERRIDE_SID;ug(a)&&(b=b||z.__1PSAPISID);if(b)return!0;var c=new rg(document);b=c.get("SAPISID")||c.get("APISID")||c.get("__Secure-3PAPISID")||c.get("SID");ug(a)&&(b=b||c.get("__Secure-1PAPISID"));return!!b}
function wg(a,b,c,d){(a=z[a])||(a=(new rg(document)).get(b));return a?ng(a,c,d):null}
function xg(a,b){b=void 0===b?!1:b;var c=lg(String(z.location.href)),d=[];if(vg(b)){c=0==c.indexOf("https:")||0==c.indexOf("chrome-extension:")||0==c.indexOf("moz-extension:");var e=c?z.__SAPISID:z.__APISID;e||(e=new rg(document),e=e.get(c?"SAPISID":"APISID")||e.get("__Secure-3PAPISID"));(e=e?ng(e,c?"SAPISIDHASH":"APISIDHASH",a):null)&&d.push(e);c&&ug(b)&&((b=wg("__1PSAPISID","__Secure-1PAPISID","SAPISID1PHASH",a))&&d.push(b),(a=wg("__3PSAPISID","__Secure-3PAPISID","SAPISID3PHASH",a))&&d.push(a))}return 0==
d.length?null:d.join(" ")}
;function yg(a){H.call(this,a,-1,zg)}
u(yg,H);var zg=[2];function Ag(a){this.h=this.i=this.j=a}
Ag.prototype.reset=function(){this.h=this.i=this.j};
Ag.prototype.getValue=function(){return this.i};function Bg(a){var b=[];Cg(new Dg,a,b);return b.join("")}
function Dg(){}
function Cg(a,b,c){if(null==b)c.push("null");else{if("object"==typeof b){if(Array.isArray(b)){var d=b;b=d.length;c.push("[");for(var e="",f=0;f<b;f++)c.push(e),Cg(a,d[f],c),e=",";c.push("]");return}if(b instanceof String||b instanceof Number||b instanceof Boolean)b=b.valueOf();else{c.push("{");e="";for(d in b)Object.prototype.hasOwnProperty.call(b,d)&&(f=b[d],"function"!=typeof f&&(c.push(e),Eg(d,c),c.push(":"),Cg(a,f,c),e=","));c.push("}");return}}switch(typeof b){case "string":Eg(b,c);break;case "number":c.push(isFinite(b)&&
!isNaN(b)?String(b):"null");break;case "boolean":c.push(String(b));break;case "function":c.push("null");break;default:throw Error("Unknown type: "+typeof b);}}}
var Fg={'"':'\\"',"\\":"\\\\","/":"\\/","\b":"\\b","\f":"\\f","\n":"\\n","\r":"\\r","\t":"\\t","\v":"\\u000b"},Gg=/\uffff/.test("\uffff")?/[\\"\x00-\x1f\x7f-\uffff]/g:/[\\"\x00-\x1f\x7f-\xff]/g;function Eg(a,b){b.push('"',a.replace(Gg,function(c){var d=Fg[c];d||(d="\\u"+(c.charCodeAt(0)|65536).toString(16).slice(1),Fg[c]=d);return d}),'"')}
;function Hg(){}
Hg.prototype.h=null;Hg.prototype.getOptions=function(){var a;(a=this.h)||(a={},Ig(this)&&(a[0]=!0,a[1]=!0),a=this.h=a);return a};var Jg;function Kg(){}
Za(Kg,Hg);function Lg(a){return(a=Ig(a))?new ActiveXObject(a):new XMLHttpRequest}
function Ig(a){if(!a.i&&"undefined"==typeof XMLHttpRequest&&"undefined"!=typeof ActiveXObject){for(var b=["MSXML2.XMLHTTP.6.0","MSXML2.XMLHTTP.3.0","MSXML2.XMLHTTP","Microsoft.XMLHTTP"],c=0;c<b.length;c++){var d=b[c];try{return new ActiveXObject(d),a.i=d}catch(e){}}throw Error("Could not create ActiveXObject. ActiveX might be disabled, or MSXML might not be installed");}return a.i}
Jg=new Kg;function Mg(a){df.call(this);this.headers=new Map;this.K=a||null;this.i=!1;this.I=this.A=null;this.m=this.S="";this.j=this.P=this.s=this.O=!1;this.o=0;this.F=null;this.ca="";this.V=this.W=!1}
Za(Mg,df);var Ng=/^https?$/i,Og=["POST","PUT"],Pg=[];function Qg(a,b,c,d,e,f,g){var h=new Mg;Pg.push(h);b&&h.ha("complete",b);h.l.add("ready",h.Kc,!0,void 0,void 0);f&&(h.o=Math.max(0,f));g&&(h.W=g);h.send(a,c,d,e)}
l=Mg.prototype;l.Kc=function(){this.dispose();kb(Pg,this)};
l.send=function(a,b,c,d){if(this.A)throw Error("[goog.net.XhrIo] Object is active with another request="+this.S+"; newUri="+a);b=b?b.toUpperCase():"GET";this.S=a;this.m="";this.O=!1;this.i=!0;this.A=this.K?Lg(this.K):Lg(Jg);this.I=this.K?this.K.getOptions():Jg.getOptions();this.A.onreadystatechange=Xa(this.ic,this);try{this.getStatus(),this.P=!0,this.A.open(b,String(a),!0),this.P=!1}catch(g){this.getStatus();Rg(this,g);return}a=c||"";c=new Map(this.headers);if(d)if(Object.getPrototypeOf(d)===Object.prototype)for(var e in d)c.set(e,
d[e]);else if("function"===typeof d.keys&&"function"===typeof d.get){e=p(d.keys());for(var f=e.next();!f.done;f=e.next())f=f.value,c.set(f,d.get(f))}else throw Error("Unknown input type for opt_headers: "+String(d));d=Array.from(c.keys()).find(function(g){return"content-type"==g.toLowerCase()});
e=z.FormData&&a instanceof z.FormData;!(0<=eb(Og,b))||d||e||c.set("Content-Type","application/x-www-form-urlencoded;charset=utf-8");b=p(c);for(d=b.next();!d.done;d=b.next())c=p(d.value),d=c.next().value,c=c.next().value,this.A.setRequestHeader(d,c);this.ca&&(this.A.responseType=this.ca);"withCredentials"in this.A&&this.A.withCredentials!==this.W&&(this.A.withCredentials=this.W);try{Ug(this),0<this.o&&(this.V=Vg(this.A),this.getStatus(),this.V?(this.A.timeout=this.o,this.A.ontimeout=Xa(this.vc,this)):
this.F=Tf(this.vc,this.o,this)),this.getStatus(),this.s=!0,this.A.send(a),this.s=!1}catch(g){this.getStatus(),Rg(this,g)}};
function Vg(a){return Ic&&"number"===typeof a.timeout&&void 0!==a.ontimeout}
l.vc=function(){"undefined"!=typeof Ma&&this.A&&(this.m="Timed out after "+this.o+"ms, aborting",this.getStatus(),ef(this,"timeout"),this.abort(8))};
function Rg(a,b){a.i=!1;a.A&&(a.j=!0,a.A.abort(),a.j=!1);a.m=b;Wg(a);Xg(a)}
function Wg(a){a.O||(a.O=!0,ef(a,"complete"),ef(a,"error"))}
l.abort=function(){this.A&&this.i&&(this.getStatus(),this.i=!1,this.j=!0,this.A.abort(),this.j=!1,ef(this,"complete"),ef(this,"abort"),Xg(this))};
l.B=function(){this.A&&(this.i&&(this.i=!1,this.j=!0,this.A.abort(),this.j=!1),Xg(this,!0));Mg.X.B.call(this)};
l.ic=function(){this.h()||(this.P||this.s||this.j?Yg(this):this.cd())};
l.cd=function(){Yg(this)};
function Yg(a){if(a.i&&"undefined"!=typeof Ma)if(a.I[1]&&4==Zg(a)&&2==a.getStatus())a.getStatus();else if(a.s&&4==Zg(a))Tf(a.ic,0,a);else if(ef(a,"readystatechange"),a.isComplete()){a.getStatus();a.i=!1;try{if($g(a))ef(a,"complete"),ef(a,"success");else{try{var b=2<Zg(a)?a.A.statusText:""}catch(c){b=""}a.m=b+" ["+a.getStatus()+"]";Wg(a)}}finally{Xg(a)}}}
function Xg(a,b){if(a.A){Ug(a);var c=a.A,d=a.I[0]?function(){}:null;
a.A=null;a.I=null;b||ef(a,"ready");try{c.onreadystatechange=d}catch(e){}}}
function Ug(a){a.A&&a.V&&(a.A.ontimeout=null);a.F&&(z.clearTimeout(a.F),a.F=null)}
l.isActive=function(){return!!this.A};
l.isComplete=function(){return 4==Zg(this)};
function $g(a){var b=a.getStatus();a:switch(b){case 200:case 201:case 202:case 204:case 206:case 304:case 1223:var c=!0;break a;default:c=!1}if(!c){if(b=0===b)a=mc(1,String(a.S)),!a&&z.self&&z.self.location&&(a=z.self.location.protocol.slice(0,-1)),b=!Ng.test(a?a.toLowerCase():"");c=b}return c}
function Zg(a){return a.A?a.A.readyState:0}
l.getStatus=function(){try{return 2<Zg(this)?this.A.status:-1}catch(a){return-1}};
l.getLastError=function(){return"string"===typeof this.m?this.m:String(this.m)};function ah(a){H.call(this,a)}
u(ah,H);function bh(a){H.call(this,a,-1,ch)}
u(bh,H);var ch=[1];var dh=["platform","platformVersion","architecture","model","uaFullVersion"];new bh;function eh(a){H.call(this,a)}
u(eh,H);function fh(a){H.call(this,a,31,gh)}
u(fh,H);var gh=[3,20,27];function hh(a){H.call(this,a,17,ih)}
u(hh,H);var ih=[3,5];function jh(a){H.call(this,a,6,kh)}
u(jh,H);var kh=[5];function lh(a){H.call(this,a)}
u(lh,H);var mh;mh=new function(a,b,c){this.i=a;this.fieldName=b;this.h=c;this.isRepeated=0;this.j=ne}(175237375,{vr:0},lh);function nh(a,b,c,d,e,f,g,h,k,m,q){df.call(this);var r=this;this.O="";this.j=[];this.Mb="";this.Nb=this.sa=-1;this.Ya=!1;this.I=this.m=null;this.F=0;this.Dc=1;this.timeoutMillis=0;this.ca=!1;df.call(this);this.Za=b||function(){};
this.s=new oh(a,f);this.Bc=d;this.Xa=q;this.Ec=Ya(jf,0,1);this.S=e||null;this.K=c||null;this.V=g||!1;this.pageId=k||null;this.logger=null;this.withCredentials=!h;this.Ha=f||!1;!this.Ha&&(65<=$b("Chromium")||45<=$b("Firefox")||12<=$b("Safari")||(Ec()||C("iPad")||C("iPod"))&&Fc());a=D(new eh,1,1);ph(this.s,a);this.o=new Ag(1E4);this.i=new Rf(this.o.getValue());ue(this,this.i);m=qh(this,m);Te(this.i,"tick",m,!1,this);this.P=new Rf(6E5);ue(this,this.P);Te(this.P,"tick",m,!1,this);this.V||this.P.start();
this.Ha||(Te(document,"visibilitychange",function(){"hidden"===document.visibilityState&&r.W()}),Te(document,"pagehide",this.W,!1,this))}
u(nh,df);function qh(a,b){return b?function(){b().then(function(){a.flush()})}:function(){a.flush()}}
nh.prototype.B=function(){this.W();df.prototype.B.call(this)};
function rh(a){a.S||(a.S=.01>a.Ec()?"https://www.google.com/log?format=json&hasfast=true":"https://play.google.com/log?format=json&hasfast=true");return a.S}
function sh(a,b){a.o=new Ag(1>b?1:b);Sf(a.i,a.o.getValue())}
nh.prototype.log=function(a){a=a.clone();var b=this.Dc++;D(a,21,b);this.O&&D(a,26,this.O);if(!Rd(a,1)){b=a;var c=Date.now().toString();D(b,1,c)}null!=Rd(a,15,!1)||D(a,15,60*(new Date).getTimezoneOffset());this.m&&(b=this.m.clone(),G(a,yg,16,b));for(;1E3<=this.j.length;)this.j.shift(),++this.F;this.j.push(a);ef(this,new th(a));this.V||this.i.enabled||this.i.start()};
nh.prototype.flush=function(a,b){var c=this;if(0===this.j.length)a&&a();else if(this.ca)uh(this);else{var d=Date.now();if(this.Nb>d&&this.sa<d)b&&b("throttled");else{var e=vh(this.s,this.j,this.F);d={};var f=this.Za();f&&(d.Authorization=f);var g=rh(this);this.K&&(d["X-Goog-AuthUser"]=this.K,g=vc(g,"authuser",this.K));this.pageId&&(d["X-Goog-PageId"]=this.pageId,g=vc(g,"pageId",this.pageId));if(f&&this.Mb===f)b&&b("stale-auth-token");else{this.j=[];this.i.enabled&&this.i.stop();this.F=0;var h=ke(e),
k;this.I&&this.I.isSupported(h.length)&&(k=this.I.compress(h));var m={url:g,body:h,Ic:1,Gb:d,requestType:"POST",withCredentials:this.withCredentials,timeoutMillis:this.timeoutMillis},q=function(t){c.o.reset();Sf(c.i,c.o.getValue());if(t){var y=null;try{var E=JSON.parse(t.replace(")]}'\n",""));y=new jh(E)}catch(F){}y&&(t=Number(ee(Rd(y,1),"-1")),0<t&&(c.sa=Date.now(),c.Nb=c.sa+t),y=mh.j(y))&&(y=ee(Rd(y,1),-1),-1!=y&&(c.Ya||sh(c,y)))}a&&a()},r=function(t,y){var E=$d(e,fh,3),F=c.o;
F.h=Math.min(3E5,2*F.h);F.i=Math.min(3E5,F.h+Math.round(.2*(Math.random()-.5)*F.h));Sf(c.i,c.o.getValue());401===t&&f&&(c.Mb=f);void 0===y&&(y=500<=t&&600>t||401===t||0===t);y&&(c.j=E.concat(c.j),c.V||c.i.enabled||c.i.start());b&&b("net-send-failed",t)},w=function(){c.Xa?c.Xa.send(m,q,r):c.Bc(m,q,r)};
k?k.then(function(t){m.Gb["Content-Encoding"]="gzip";m.Gb["Content-Type"]="application/binary";m.body=t;m.Ic=2;w()},function(){w()}):w()}}}};
nh.prototype.W=function(){this.flush()};
function uh(a){wh(a,function(b,c){b=vc(b,"format","json");b=window.navigator.sendBeacon(b,ke(c));a.ca&&!b&&(a.ca=!1);return b})}
function wh(a,b){if(0!==a.j.length){var c=Cc(rh(a),"format");c=tc(c,"auth",a.Za(),"authuser",a.K||"0");for(var d=0;10>d&&a.j.length;++d){var e=a.j.slice(0,32),f=vh(a.s,e,a.F);if(!b(c,f))break;a.F=0;a.j=a.j.slice(e.length)}a.i.enabled&&a.i.stop()}}
function th(){we.call(this,"event-logged",void 0)}
u(th,we);function oh(a,b){this.i=b=void 0===b?!1:b;this.uach=this.locale=null;this.h=new hh;D(this.h,2,a);b||(this.locale=document.documentElement.getAttribute("lang"));ph(this,new eh)}
function ph(a,b){G(a.h,eh,1,b);Rd(b,1)||D(b,1,1);a.i||(b=xh(a),Rd(b,5)||D(b,5,a.locale));a.uach&&(b=xh(a),Xd(b,bh,9)||G(b,bh,9,a.uach))}
function yh(a,b){var c=void 0===c?dh:c;b(window,c).then(function(d){a.uach=d;d=xh(a);G(d,bh,9,a.uach);return!0}).catch(function(){return!1})}
function xh(a){a=Xd(a.h,eh,1);var b=Xd(a,ah,11);b||(b=new ah,G(a,ah,11,b));return b}
function vh(a,b,c){c=void 0===c?0:c;a=a.h.clone();var d=Date.now().toString();a=D(a,4,d);b=be(a,fh,3,b);c&&D(b,14,c);return b}
;function zh(a,b,c){Qg(a.url,function(d){d=d.target;if($g(d)){try{var e=d.A?d.A.responseText:""}catch(f){e=""}b(e)}else c(d.getStatus())},a.requestType,a.body,a.Gb,a.timeoutMillis,a.withCredentials)}
;function Ah(){this.j="https://play.google.com/log?format=json&hasfast=true";this.s=!1;this.m=zh;this.h=""}
;function Bh(){var a=void 0===a?"":a;var b=void 0===b?"":b;var c=new Ah;c.h="";""!=a&&(c.j=a);b&&(c.i=b);a=new nh(1828,c.I?c.I:xg,"0",c.m,c.j,c.s,!1,c.P,void 0,void 0,c.o?c.o:void 0);c.M&&ph(a.s,c.M);if(c.i){b=c.i;var d=xh(a.s);D(d,7,b)}c.l&&(a.I=c.l);c.h&&(a.O=c.h);c.v&&((b=c.v)?(a.m||(a.m=new yg),b=ke(b),D(a.m,4,b)):a.m&&D(a.m,4,void 0,!1));c.K&&(d=c.K,a.m||(a.m=new yg),b=a.m,d=null==d?Dd:zd(d,1),D(b,2,d));c.F&&(b=c.F,a.Ya=!0,sh(a,b));c.O&&yh(a.s,c.O);this.h=a}
Bh.prototype.flush=function(a){var b=a||[];if(b.length){a=new jg;for(var c=[],d=0;d<b.length;d++){var e=b[d],f=e,g=new bg;var h=D(g,1,f.i);var k=f;g=[];for(var m=0;m<k.h.length;m++)g.push(k.h[m].La);if(null==g)g=Dd;else{for(k=0;k<g.length;k++);g=zd(g,5)}g=D(h,3,g);h=[];k=[];m=p(f.Aa.keys());for(var q=m.next();!q.done;q=m.next())k.push(q.value.split(","));for(m=0;m<k.length;m++){q=k[m];var r=f.l;for(var w=f.ub(q)||[],t=[],y=0;y<w.length;y++){var E=w[y];E=E&&E.Sb;var F=new gg;switch(r){case 3:Vd(F,
1,ig,Number(E));break;case 2:Vd(F,2,ig,Number(E))}t.push(F)}r=t;for(w=0;w<r.length;w++){t=r[w];y=new dg;t=G(y,gg,2,t);y=q;E=[];F=f;for(var O=[],N=0;N<F.h.length;N++)O.push(F.h[N].Ma);F=O;for(O=0;O<F.length;O++){N=F[O];var R=y[O],ca=new fg;switch(N){case 3:Vd(ca,1,hg,String(R));break;case 2:Vd(ca,2,hg,Number(R));break;case 1:Vd(ca,3,hg,"true"==R)}E.push(ca)}be(t,fg,1,E);h.push(t)}}be(g,dg,4,h);c.push(g);e.clear()}be(a,bg,1,c);b=this.h;a instanceof fh?b.log(a):(c=new fh,a=ke(a),a=D(c,8,a),b.log(a));
this.h.flush()}};function Ch(a){this.wa=a;this.v=Dh();this.l=new Bh;this.h=new Uf(this.l);this.o=new Zf(this.h);this.i=new $f(this.h);this.j=new ag(this.h);this.m=new Yf(this.h)}
function Dh(){var a,b,c;return null!=(c=null==(a=globalThis.performance)?void 0:null==(b=a.now)?void 0:b.call(a))?c:Date.now()}
;function Eh(){var a=this;this.promise=new Promise(function(b,c){a.resolve=b;a.reject=c})}
;function Fh(a){var b=this;this.i=!1;var c=a.program;var d=a.Tc;if(a.gd){var e;this.la=null!=(e=a.la)?e:new Ch(d)}var f=new Eh;this.j=f.promise;if(!z[d]){var g;null!=(g=this.la)&&g.m.h.rb("/client_streamz/bg/fiec",g.wa,1)}this.l=p((0,z[d].a)(c,function(h,k){Promise.resolve().then(function(){var m;if(null!=(m=b.la)){var q=Dh()-m.v;m.o.h.sb("/client_streamz/bg/fil",q,m.wa)}f.resolve({Gc:h,ud:k})})},!0)).next().value;
this.td=f.promise.then(function(){})}
Fh.prototype.snapshot=function(a){var b=this;if(this.i)throw Error("Already disposed");var c=Dh(),d;null!=(d=this.la)&&d.i.h.rb("/client_streamz/bg/fsc",d.wa);return this.j.then(function(e){var f=e.Gc;return new Promise(function(g){f(function(h){var k;if(null!=(k=b.la)){var m=Dh()-c;k.j.h.sb("/client_streamz/bg/fsl",m,k.wa)}g(h)},[a.Ub,
a.vd])})})};
Fh.prototype.tc=function(a){if(this.i)throw Error("Already disposed");var b=Dh(),c;null!=(c=this.la)&&c.i.h.rb("/client_streamz/bg/fsc",c.wa);a=this.l([a.Ub,a.vd]);null!=(c=this.la)&&(b=Dh()-b,c.j.h.sb("/client_streamz/bg/fsl",b,c.wa));return a};
Fh.prototype.dispose=function(){var a;null!=(a=this.la)&&a.h.tb();this.i=!0;this.j.then(function(b){(b=b.ud)&&b()})};
Fh.prototype.h=function(){return this.i};var Gh=window;Cb("csi.gstatic.com");Cb("googleads.g.doubleclick.net");Cb("partner.googleadservices.com");Cb("pubads.g.doubleclick.net");Cb("securepubads.g.doubleclick.net");Cb("tpc.googlesyndication.com");/*

 SPDX-License-Identifier: Apache-2.0
*/
var Hh;try{new URL("s://g"),Hh=!0}catch(a){Hh=!1}var Ih=Hh;function Jh(a,b){a.src=Hb(b);var c,d;(c=(b=null==(d=(c=(a.ownerDocument&&a.ownerDocument.defaultView||window).document).querySelector)?void 0:d.call(c,"script[nonce]"))?b.nonce||b.getAttribute("nonce")||"":"")&&a.setAttribute("nonce",c)}
;function Kh(a){this.Yc=a}
function Lh(a){return new Kh(function(b){return b.substr(0,a.length+1).toLowerCase()===a+":"})}
var Mh=[Lh("data"),Lh("http"),Lh("https"),Lh("mailto"),Lh("ftp"),new Kh(function(a){return/^[^:]*([/?#]|$)/.test(a)})];function Nh(a){var b=Oh;if(b)for(var c in b)Object.prototype.hasOwnProperty.call(b,c)&&a(b[c],c,b)}
function Ph(){var a=[];Nh(function(b){a.push(b)});
return a}
var Oh={Pd:"allow-forms",Qd:"allow-modals",Rd:"allow-orientation-lock",Sd:"allow-pointer-lock",Td:"allow-popups",Ud:"allow-popups-to-escape-sandbox",Vd:"allow-presentation",Wd:"allow-same-origin",Xd:"allow-scripts",Yd:"allow-top-navigation",Zd:"allow-top-navigation-by-user-activation"},Qh=db(function(){return Ph()});
function Rh(){var a=Sh(),b={};fb(Qh(),function(c){a.sandbox&&a.sandbox.supports&&a.sandbox.supports(c)&&(b[c]=!0)});
return b}
function Sh(){var a=void 0===a?document:a;return a.createElement("iframe")}
;function Th(a){"number"==typeof a&&(a=Math.round(a)+"px");return a}
;var Uh=(new Date).getTime();var Vh="client_dev_domain client_dev_regex_map client_dev_root_url client_rollout_override expflag forcedCapability jsfeat jsmode mods".split(" ");ia(Vh);function Wh(a){df.call(this);var b=this;this.s=this.j=0;this.Y=null!=a?a:{R:function(e,f){return setTimeout(e,f)},
ea:function(e){clearTimeout(e)}};
var c,d;this.i=null!=(d=null==(c=window.navigator)?void 0:c.onLine)?d:!0;this.m=function(){return x(function(e){return v(e,Xh(b),0)})};
window.addEventListener("offline",this.m);window.addEventListener("online",this.m);this.s||Yh(this)}
u(Wh,df);function Zh(){var a=$h;Wh.h||(Wh.h=new Wh(a));return Wh.h}
Wh.prototype.dispose=function(){window.removeEventListener("offline",this.m);window.removeEventListener("online",this.m);this.Y.ea(this.s);delete Wh.h};
Wh.prototype.U=function(){return this.i};
function Yh(a){a.s=a.Y.R(function(){var b;return x(function(c){if(1==c.h)return a.i?(null==(b=window.navigator)?0:b.onLine)?c.u(3):v(c,Xh(a),3):v(c,Xh(a),3);Yh(a);c.h=0})},3E4)}
function Xh(a,b){return a.o?a.o:a.o=new Promise(function(c){var d,e,f,g;return x(function(h){switch(h.h){case 1:return d=window.AbortController?new window.AbortController:void 0,f=null==(e=d)?void 0:e.signal,g=!1,ya(h,2,3),d&&(a.j=a.Y.R(function(){d.abort()},b||2E4)),v(h,fetch("/generate_204",{method:"HEAD",
signal:f}),5);case 5:g=!0;case 3:Ba(h);a.o=void 0;a.j&&(a.Y.ea(a.j),a.j=0);g!==a.i&&(a.i=g,a.i?ef(a,"networkstatus-online"):ef(a,"networkstatus-offline"));c(g);Ca(h);break;case 2:Aa(h),g=!1,h.u(3)}})})}
;function ai(){this.data_=[];this.h=-1}
ai.prototype.set=function(a,b){b=void 0===b?!0:b;0<=a&&52>a&&Number.isInteger(a)&&this.data_[a]!==b&&(this.data_[a]=b,this.h=-1)};
ai.prototype.get=function(a){return!!this.data_[a]};
function bi(a){-1===a.h&&(a.h=ib(a.data_,function(b,c,d){return c?b+Math.pow(2,d):b},0));
return a.h}
;function ci(a,b){this.h=a[z.Symbol.iterator]();this.i=b}
ci.prototype[Symbol.iterator]=function(){return this};
ci.prototype.next=function(){var a=this.h.next();return{value:a.done?void 0:this.i.call(void 0,a.value),done:a.done}};
function di(a,b){return new ci(a,b)}
;function ei(){this.blockSize=-1}
;function fi(){this.blockSize=-1;this.blockSize=64;this.h=[];this.m=[];this.o=[];this.j=[];this.j[0]=128;for(var a=1;a<this.blockSize;++a)this.j[a]=0;this.l=this.i=0;this.reset()}
Za(fi,ei);fi.prototype.reset=function(){this.h[0]=1732584193;this.h[1]=4023233417;this.h[2]=2562383102;this.h[3]=271733878;this.h[4]=3285377520;this.l=this.i=0};
function gi(a,b,c){c||(c=0);var d=a.o;if("string"===typeof b)for(var e=0;16>e;e++)d[e]=b.charCodeAt(c)<<24|b.charCodeAt(c+1)<<16|b.charCodeAt(c+2)<<8|b.charCodeAt(c+3),c+=4;else for(e=0;16>e;e++)d[e]=b[c]<<24|b[c+1]<<16|b[c+2]<<8|b[c+3],c+=4;for(e=16;80>e;e++){var f=d[e-3]^d[e-8]^d[e-14]^d[e-16];d[e]=(f<<1|f>>>31)&4294967295}b=a.h[0];c=a.h[1];var g=a.h[2],h=a.h[3],k=a.h[4];for(e=0;80>e;e++){if(40>e)if(20>e){f=h^c&(g^h);var m=1518500249}else f=c^g^h,m=1859775393;else 60>e?(f=c&g|h&(c|g),m=2400959708):
(f=c^g^h,m=3395469782);f=(b<<5|b>>>27)+f+k+m+d[e]&4294967295;k=h;h=g;g=(c<<30|c>>>2)&4294967295;c=b;b=f}a.h[0]=a.h[0]+b&4294967295;a.h[1]=a.h[1]+c&4294967295;a.h[2]=a.h[2]+g&4294967295;a.h[3]=a.h[3]+h&4294967295;a.h[4]=a.h[4]+k&4294967295}
fi.prototype.update=function(a,b){if(null!=a){void 0===b&&(b=a.length);for(var c=b-this.blockSize,d=0,e=this.m,f=this.i;d<b;){if(0==f)for(;d<=c;)gi(this,a,d),d+=this.blockSize;if("string"===typeof a)for(;d<b;){if(e[f]=a.charCodeAt(d),++f,++d,f==this.blockSize){gi(this,e);f=0;break}}else for(;d<b;)if(e[f]=a[d],++f,++d,f==this.blockSize){gi(this,e);f=0;break}}this.i=f;this.l+=b}};
fi.prototype.digest=function(){var a=[],b=8*this.l;56>this.i?this.update(this.j,56-this.i):this.update(this.j,this.blockSize-(this.i-56));for(var c=this.blockSize-1;56<=c;c--)this.m[c]=b&255,b/=256;gi(this,this.m);for(c=b=0;5>c;c++)for(var d=24;0<=d;d-=8)a[b]=this.h[c]>>d&255,++b;return a};function hi(a){return"string"==typeof a.className?a.className:a.getAttribute&&a.getAttribute("class")||""}
function ii(a,b){"string"==typeof a.className?a.className=b:a.setAttribute&&a.setAttribute("class",b)}
function ji(a,b){a.classList?b=a.classList.contains(b):(a=a.classList?a.classList:hi(a).match(/\S+/g)||[],b=0<=eb(a,b));return b}
function ki(){var a=document.body;a.classList?a.classList.remove("inverted-hdpi"):ji(a,"inverted-hdpi")&&ii(a,Array.prototype.filter.call(a.classList?a.classList:hi(a).match(/\S+/g)||[],function(b){return"inverted-hdpi"!=b}).join(" "))}
;function li(){}
li.prototype.next=function(){return mi};
var mi={done:!0,value:void 0};function ni(a){return{value:a,done:!1}}
li.prototype.da=function(){return this};function oi(a){if(a instanceof pi||a instanceof qi||a instanceof ri)return a;if("function"==typeof a.next)return new pi(function(){return a});
if("function"==typeof a[Symbol.iterator])return new pi(function(){return a[Symbol.iterator]()});
if("function"==typeof a.da)return new pi(function(){return a.da()});
throw Error("Not an iterator or iterable.");}
function pi(a){this.i=a}
pi.prototype.da=function(){return new qi(this.i())};
pi.prototype[Symbol.iterator]=function(){return new ri(this.i())};
pi.prototype.h=function(){return new ri(this.i())};
function qi(a){this.i=a}
u(qi,li);qi.prototype.next=function(){return this.i.next()};
qi.prototype[Symbol.iterator]=function(){return new ri(this.i)};
qi.prototype.h=function(){return new ri(this.i)};
function ri(a){pi.call(this,function(){return a});
this.j=a}
u(ri,pi);ri.prototype.next=function(){return this.j.next()};function si(a,b){this.i={};this.h=[];this.oa=this.size=0;var c=arguments.length;if(1<c){if(c%2)throw Error("Uneven number of arguments");for(var d=0;d<c;d+=2)this.set(arguments[d],arguments[d+1])}else if(a)if(a instanceof si)for(c=a.xb(),d=0;d<c.length;d++)this.set(c[d],a.get(c[d]));else for(d in a)this.set(d,a[d])}
l=si.prototype;l.xb=function(){ti(this);return this.h.concat()};
l.has=function(a){return ui(this.i,a)};
l.equals=function(a,b){if(this===a)return!0;if(this.size!=a.size)return!1;b=b||vi;ti(this);for(var c,d=0;c=this.h[d];d++)if(!b(this.get(c),a.get(c)))return!1;return!0};
function vi(a,b){return a===b}
l.Oa=function(){return 0==this.size};
l.clear=function(){this.i={};this.oa=this.size=this.h.length=0};
l.remove=function(a){return this.delete(a)};
l.delete=function(a){return ui(this.i,a)?(delete this.i[a],--this.size,this.oa++,this.h.length>2*this.size&&ti(this),!0):!1};
function ti(a){if(a.size!=a.h.length){for(var b=0,c=0;b<a.h.length;){var d=a.h[b];ui(a.i,d)&&(a.h[c++]=d);b++}a.h.length=c}if(a.size!=a.h.length){var e={};for(c=b=0;b<a.h.length;)d=a.h[b],ui(e,d)||(a.h[c++]=d,e[d]=1),b++;a.h.length=c}}
l.get=function(a,b){return ui(this.i,a)?this.i[a]:b};
l.set=function(a,b){ui(this.i,a)||(this.size+=1,this.h.push(a),this.oa++);this.i[a]=b};
l.forEach=function(a,b){for(var c=this.xb(),d=0;d<c.length;d++){var e=c[d],f=this.get(e);a.call(b,f,e,this)}};
l.clone=function(){return new si(this)};
l.keys=function(){return oi(this.da(!0)).h()};
l.values=function(){return oi(this.da(!1)).h()};
l.entries=function(){var a=this;return di(this.keys(),function(b){return[b,a.get(b)]})};
l.da=function(a){ti(this);var b=0,c=this.oa,d=this,e=new li;e.next=function(){if(c!=d.oa)throw Error("The map has changed since the iterator was created");if(b>=d.h.length)return mi;var f=d.h[b++];return ni(a?f:d.i[f])};
return e};
function ui(a,b){return Object.prototype.hasOwnProperty.call(a,b)}
;function K(a){J.call(this);this.o=1;this.l=[];this.m=0;this.i=[];this.j={};this.s=!!a}
Za(K,J);l=K.prototype;l.subscribe=function(a,b,c){var d=this.j[a];d||(d=this.j[a]=[]);var e=this.o;this.i[e]=a;this.i[e+1]=b;this.i[e+2]=c;this.o=e+3;d.push(e);return e};
function wi(a,b,c,d){if(b=a.j[b]){var e=a.i;(b=b.find(function(f){return e[f+1]==c&&e[f+2]==d}))&&a.Ga(b)}}
l.Ga=function(a){var b=this.i[a];if(b){var c=this.j[b];0!=this.m?(this.l.push(a),this.i[a+1]=function(){}):(c&&kb(c,a),delete this.i[a],delete this.i[a+1],delete this.i[a+2])}return!!b};
l.ra=function(a,b){var c=this.j[a];if(c){for(var d=Array(arguments.length-1),e=1,f=arguments.length;e<f;e++)d[e-1]=arguments[e];if(this.s)for(e=0;e<c.length;e++){var g=c[e];xi(this.i[g+1],this.i[g+2],d)}else{this.m++;try{for(e=0,f=c.length;e<f&&!this.h();e++)g=c[e],this.i[g+1].apply(this.i[g+2],d)}finally{if(this.m--,0<this.l.length&&0==this.m)for(;c=this.l.pop();)this.Ga(c)}}return 0!=e}return!1};
function xi(a,b,c){xf(function(){a.apply(b,c)})}
l.clear=function(a){if(a){var b=this.j[a];b&&(b.forEach(this.Ga,this),delete this.j[a])}else this.i.length=0,this.j={}};
l.B=function(){K.X.B.call(this);this.clear();this.l.length=0};function yi(a){this.h=a}
yi.prototype.set=function(a,b){void 0===b?this.h.remove(a):this.h.set(a,Bg(b))};
yi.prototype.get=function(a){try{var b=this.h.get(a)}catch(c){return}if(null!==b)try{return JSON.parse(b)}catch(c){throw"Storage: Invalid value was encountered";}};
yi.prototype.remove=function(a){this.h.remove(a)};function zi(a){this.h=a}
Za(zi,yi);function Ai(a){this.data=a}
function Bi(a){return void 0===a||a instanceof Ai?a:new Ai(a)}
zi.prototype.set=function(a,b){zi.X.set.call(this,a,Bi(b))};
zi.prototype.i=function(a){a=zi.X.get.call(this,a);if(void 0===a||a instanceof Object)return a;throw"Storage: Invalid value was encountered";};
zi.prototype.get=function(a){if(a=this.i(a)){if(a=a.data,void 0===a)throw"Storage: Invalid value was encountered";}else a=void 0;return a};function Ci(a){this.h=a}
Za(Ci,zi);Ci.prototype.set=function(a,b,c){if(b=Bi(b)){if(c){if(c<Date.now()){Ci.prototype.remove.call(this,a);return}b.expiration=c}b.creation=Date.now()}Ci.X.set.call(this,a,b)};
Ci.prototype.i=function(a){var b=Ci.X.i.call(this,a);if(b){var c=b.creation,d=b.expiration;if(d&&d<Date.now()||c&&c>Date.now())Ci.prototype.remove.call(this,a);else return b}};function Di(){}
;function Ei(){}
Za(Ei,Di);Ei.prototype[Symbol.iterator]=function(){return oi(this.da(!0)).h()};
Ei.prototype.clear=function(){var a=Array.from(this);a=p(a);for(var b=a.next();!b.done;b=a.next())this.remove(b.value)};function Fi(a){this.h=a}
Za(Fi,Ei);l=Fi.prototype;l.isAvailable=function(){if(!this.h)return!1;try{return this.h.setItem("__sak","1"),this.h.removeItem("__sak"),!0}catch(a){return!1}};
l.set=function(a,b){try{this.h.setItem(a,b)}catch(c){if(0==this.h.length)throw"Storage mechanism: Storage disabled";throw"Storage mechanism: Quota exceeded";}};
l.get=function(a){a=this.h.getItem(a);if("string"!==typeof a&&null!==a)throw"Storage mechanism: Invalid value was encountered";return a};
l.remove=function(a){this.h.removeItem(a)};
l.da=function(a){var b=0,c=this.h,d=new li;d.next=function(){if(b>=c.length)return mi;var e=c.key(b++);if(a)return ni(e);e=c.getItem(e);if("string"!==typeof e)throw"Storage mechanism: Invalid value was encountered";return ni(e)};
return d};
l.clear=function(){this.h.clear()};
l.key=function(a){return this.h.key(a)};function Gi(){var a=null;try{a=window.localStorage||null}catch(b){}this.h=a}
Za(Gi,Fi);function Hi(a,b){this.i=a;this.h=null;var c;if(c=Ic)c=!(9<=Number(Vc));if(c){Ii||(Ii=new si);this.h=Ii.get(a);this.h||(b?this.h=document.getElementById(b):(this.h=document.createElement("userdata"),this.h.addBehavior("#default#userData"),document.body.appendChild(this.h)),Ii.set(a,this.h));try{this.h.load(this.i)}catch(d){this.h=null}}}
Za(Hi,Ei);var Ji={".":".2E","!":".21","~":".7E","*":".2A","'":".27","(":".28",")":".29","%":"."},Ii=null;function Ki(a){return"_"+encodeURIComponent(a).replace(/[.!~*'()%]/g,function(b){return Ji[b]})}
l=Hi.prototype;l.isAvailable=function(){return!!this.h};
l.set=function(a,b){this.h.setAttribute(Ki(a),b);Li(this)};
l.get=function(a){a=this.h.getAttribute(Ki(a));if("string"!==typeof a&&null!==a)throw"Storage mechanism: Invalid value was encountered";return a};
l.remove=function(a){this.h.removeAttribute(Ki(a));Li(this)};
l.da=function(a){var b=0,c=this.h.XMLDocument.documentElement.attributes,d=new li;d.next=function(){if(b>=c.length)return mi;var e=c[b++];if(a)return ni(decodeURIComponent(e.nodeName.replace(/\./g,"%")).slice(1));e=e.nodeValue;if("string"!==typeof e)throw"Storage mechanism: Invalid value was encountered";return ni(e)};
return d};
l.clear=function(){for(var a=this.h.XMLDocument.documentElement,b=a.attributes.length;0<b;b--)a.removeAttribute(a.attributes[b-1].nodeName);Li(this)};
function Li(a){try{a.h.save(a.i)}catch(b){throw"Storage mechanism: Quota exceeded";}}
;function Mi(a,b){this.i=a;this.h=b+"::"}
Za(Mi,Ei);Mi.prototype.set=function(a,b){this.i.set(this.h+a,b)};
Mi.prototype.get=function(a){return this.i.get(this.h+a)};
Mi.prototype.remove=function(a){this.i.remove(this.h+a)};
Mi.prototype.da=function(a){var b=this.i[Symbol.iterator](),c=this,d=new li;d.next=function(){var e=b.next();if(e.done)return e;for(e=e.value;e.slice(0,c.h.length)!=c.h;){e=b.next();if(e.done)return e;e=e.value}return ni(a?e.slice(c.h.length):c.i.get(e))};
return d};function Ni(a){this.name=a}
;var Oi=new Ni("rawColdConfigGroup");var Pi=new Ni("rawHotConfigGroup");function Qi(a){H.call(this,a)}
u(Qi,H);function Ri(a){H.call(this,a)}
u(Ri,H);Ri.prototype.getKey=function(){return Rd(this,1)};
Ri.prototype.getValue=function(){return de(this,2===Wd(this,Si)?2:-1)};
var Si=[2,3,4,5,6];function Ti(a){H.call(this,a)}
u(Ti,H);function Ui(a){H.call(this,a)}
u(Ui,H);function Vi(a){H.call(this,a,-1,Wi)}
u(Vi,H);var Wi=[2];function Xi(a){H.call(this,a,-1,Yi)}
u(Xi,H);Xi.prototype.getPlayerType=function(){return Rd(this,36)};
Xi.prototype.setHomeGroupInfo=function(a){return G(this,Vi,81,a)};
Xi.prototype.clearLocationPlayabilityToken=function(){return D(this,89,void 0,!1)};
var Yi=[9,66,24,32,86,100,101];function Zi(a){H.call(this,a,-1,$i)}
u(Zi,H);var $i=[15,26,28];function aj(a){H.call(this,a,-1,bj)}
u(aj,H);var bj=[5];function cj(a){H.call(this,a)}
u(cj,H);function dj(a){H.call(this,a,-1,ej)}
u(dj,H);dj.prototype.setSafetyMode=function(a){return D(this,5,a)};
var ej=[12];function fj(a){H.call(this,a,-1,gj)}
u(fj,H);fj.prototype.i=function(a){return G(this,Xi,1,a)};
var gj=[12];var hj=new Ni("continuationCommand");var ij=new Ni("webCommandMetadata");var jj=new Ni("signalServiceEndpoint");var kj={oi:"EMBEDDED_PLAYER_MODE_UNKNOWN",li:"EMBEDDED_PLAYER_MODE_DEFAULT",ni:"EMBEDDED_PLAYER_MODE_PFP",mi:"EMBEDDED_PLAYER_MODE_PFL"};var lj=new Ni("feedbackEndpoint");var mj={Oq:"WEB_DISPLAY_MODE_UNKNOWN",Kq:"WEB_DISPLAY_MODE_BROWSER",Mq:"WEB_DISPLAY_MODE_MINIMAL_UI",Nq:"WEB_DISPLAY_MODE_STANDALONE",Lq:"WEB_DISPLAY_MODE_FULLSCREEN"};function nj(a){H.call(this,a,-1,oj)}
u(nj,H);function pj(a){H.call(this,a)}
u(pj,H);pj.prototype.getKey=function(){return ee(Rd(this,1),"")};
pj.prototype.getValue=function(){return ee(Rd(this,2),"")};
var oj=[4,5];function qj(a){H.call(this,a)}
u(qj,H);function rj(a){H.call(this,a)}
u(rj,H);var sj=[2,3,4];function tj(a){H.call(this,a)}
u(tj,H);tj.prototype.getMessage=function(){return ee(Rd(this,1),"")};function uj(a){H.call(this,a)}
u(uj,H);function vj(a){H.call(this,a)}
u(vj,H);function wj(a){H.call(this,a,-1,xj)}
u(wj,H);var xj=[10,17];function yj(a){H.call(this,a)}
u(yj,H);function zj(a){H.call(this,a)}
u(zj,H);function Aj(a){H.call(this,a)}
u(Aj,H);function Bj(a){H.call(this,a)}
u(Bj,H);function Cj(a){H.call(this,a)}
u(Cj,H);function Dj(a,b){G(a,Aj,1,b)}
Cj.prototype.i=function(a){D(this,2,a)};
function Ej(a){H.call(this,a)}
u(Ej,H);function Fj(a,b){G(a,Aj,1,b)}
;function Gj(a){H.call(this,a,-1,Hj)}
u(Gj,H);Gj.prototype.i=function(a){D(this,1,a)};
function Ij(a,b){G(a,Aj,2,b)}
var Hj=[3];function Jj(a){H.call(this,a)}
u(Jj,H);Jj.prototype.i=function(a){D(this,1,a)};function Kj(a){H.call(this,a)}
u(Kj,H);Kj.prototype.i=function(a){D(this,1,a)};function Lj(a){H.call(this,a)}
u(Lj,H);Lj.prototype.i=function(a){D(this,1,a)};function Mj(a){H.call(this,a)}
u(Mj,H);Mj.prototype.i=function(a){D(this,1,a)};function Nj(a){H.call(this,a)}
u(Nj,H);function Oj(a){H.call(this,a)}
u(Oj,H);function Pj(a){H.call(this,a,-1,Qj)}
u(Pj,H);Pj.prototype.getPlayerType=function(){var a=Rd(this,7);return null==a?0:a};
Pj.prototype.setVideoId=function(a){return D(this,19,a)};
function Rj(a,b){ce(a,68,Sj,b)}
function Sj(a){H.call(this,a)}
u(Sj,H);Sj.prototype.getId=function(){return ee(Rd(this,2),"")};
var Qj=[83,68];function Tj(a){H.call(this,a)}
u(Tj,H);function Uj(a){H.call(this,a)}
u(Uj,H);function Vj(a){H.call(this,a)}
u(Vj,H);function Wj(a){H.call(this,a,459)}
u(Wj,H);
var Xj=[23,24,11,6,7,5,2,3,13,20,21,22,28,32,37,229,241,45,59,225,288,72,73,78,208,156,202,215,74,76,79,80,111,85,91,97,100,102,105,119,126,127,136,146,148,151,157,158,159,163,164,168,444,176,222,383,177,178,179,458,411,184,188,189,190,191,193,194,195,196,197,198,199,200,201,402,320,203,204,205,206,258,259,260,261,327,209,219,226,227,232,233,234,240,244,247,248,249,251,256,257,266,254,255,270,272,278,291,293,300,304,308,309,310,311,313,314,319,321,323,324,328,330,331,332,334,337,338,340,344,348,350,
351,352,353,354,355,356,357,358,361,363,364,368,369,370,373,374,375,378,380,381,388,389,403,410,412,429,413,414,415,416,417,418,430,423,424,425,426,427,431,117,439,441,448];var Yj={qj:0,bj:1,hj:2,ij:4,nj:8,jj:16,kj:32,pj:64,oj:128,dj:256,fj:512,mj:1024,ej:2048,gj:4096,cj:8192,lj:16384};function Zj(a){H.call(this,a)}
u(Zj,H);function ak(a){H.call(this,a)}
u(ak,H);ak.prototype.setVideoId=function(a){return Vd(this,1,bk,a)};
ak.prototype.getPlaylistId=function(){return de(this,2===Wd(this,bk)?2:-1)};
var bk=[1,2];function ck(a){H.call(this,a,-1,dk)}
u(ck,H);var dk=[3];var ek=new Ni("webPlayerShareEntityServiceEndpoint");var fk=new Ni("playlistEditEndpoint");var Ak=new Ni("modifyChannelNotificationPreferenceEndpoint");var Bk=new Ni("unsubscribeEndpoint");var Ck=new Ni("subscribeEndpoint");function Dk(a,b){1<b.length?a[b[0]]=b[1]:1===b.length&&Object.assign(a,b[0])}
;var Ek=z.window,Fk,Gk,Hk=(null==Ek?void 0:null==(Fk=Ek.yt)?void 0:Fk.config_)||(null==Ek?void 0:null==(Gk=Ek.ytcfg)?void 0:Gk.data_)||{};A("yt.config_",Hk);function Ik(){Dk(Hk,arguments)}
function L(a,b){return a in Hk?Hk[a]:b}
function Jk(){var a=Hk.EXPERIMENT_FLAGS;return a?a.web_disable_gel_stp_ecatcher_killswitch:void 0}
;function M(a){a=Kk(a);return"string"===typeof a&&"false"===a?!1:!!a}
function Lk(a,b){a=Kk(a);return void 0===a&&void 0!==b?b:Number(a||0)}
function Mk(){return L("EXPERIMENTS_TOKEN","")}
function Kk(a){var b=L("EXPERIMENTS_FORCED_FLAGS",{})||{};return void 0!==b[a]?b[a]:L("EXPERIMENT_FLAGS",{})[a]}
function Nk(){for(var a=[],b=L("EXPERIMENTS_FORCED_FLAGS",{}),c=p(Object.keys(b)),d=c.next();!d.done;d=c.next())d=d.value,a.push({key:d,value:String(b[d])});c=L("EXPERIMENT_FLAGS",{});var e=p(Object.keys(c));for(d=e.next();!d.done;d=e.next())d=d.value,d.startsWith("force_")&&void 0===b[d]&&a.push({key:d,value:String(c[d])});return a}
;var Ok=[];function Pk(a){Ok.forEach(function(b){return b(a)})}
function Qk(a){return a&&window.yterr?function(){try{return a.apply(this,arguments)}catch(b){Rk(b)}}:a}
function Rk(a,b,c,d){var e=B("yt.logging.errors.log");e?e(a,"ERROR",b,c,d):(e=L("ERRORS",[]),e.push([a,"ERROR",b,c,d]),Ik("ERRORS",e));Pk(a)}
function Sk(a,b,c,d){var e=B("yt.logging.errors.log");e?e(a,"WARNING",b,c,d):(e=L("ERRORS",[]),e.push([a,"WARNING",b,c,d]),Ik("ERRORS",e))}
;function Tk(){var a=Uk;B("yt.ads.biscotti.getId_")||A("yt.ads.biscotti.getId_",a)}
function Vk(a){A("yt.ads.biscotti.lastId_",a)}
;var Wk=/^[\w.]*$/,Xk={q:!0,search_query:!0};function Yk(a,b){b=a.split(b);for(var c={},d=0,e=b.length;d<e;d++){var f=b[d].split("=");if(1==f.length&&f[0]||2==f.length)try{var g=Zk(f[0]||""),h=Zk(f[1]||"");g in c?Array.isArray(c[g])?lb(c[g],h):c[g]=[c[g],h]:c[g]=h}catch(r){var k=r,m=f[0],q=String(Yk);k.args=[{key:m,value:f[1],query:a,method:$k==q?"unchanged":q}];Xk.hasOwnProperty(m)||Sk(k)}}return c}
var $k=String(Yk);function al(a){var b=[];mb(a,function(c,d){var e=encodeURIComponent(String(d)),f;Array.isArray(c)?f=c:f=[c];fb(f,function(g){""==g?b.push(e):b.push(e+"="+encodeURIComponent(String(g)))})});
return b.join("&")}
function bl(a){"?"==a.charAt(0)&&(a=a.substr(1));return Yk(a,"&")}
function cl(a){return-1!=a.indexOf("?")?(a=(a||"").split("#")[0],a=a.split("?",2),bl(1<a.length?a[1]:a[0])):{}}
function dl(a,b,c){var d=a.split("#",2);a=d[0];d=1<d.length?"#"+d[1]:"";var e=a.split("?",2);a=e[0];e=bl(e[1]||"");for(var f in b)!c&&null!==e&&f in e||(e[f]=b[f]);return uc(a,e)+d}
function el(a){if(!b)var b=window.location.href;var c=mc(1,a),d=nc(a);c&&d?(a=a.match(kc),b=b.match(kc),a=a[3]==b[3]&&a[1]==b[1]&&a[4]==b[4]):a=d?nc(b)==d&&(Number(mc(4,b))||null)==(Number(mc(4,a))||null):!0;return a}
function Zk(a){return a&&a.match(Wk)?a:decodeURIComponent(a.replace(/\+/g," "))}
;function fl(a){var b=gl;a=void 0===a?B("yt.ads.biscotti.lastId_")||"":a;var c=Object,d=c.assign,e={};e.dt=Uh;e.flash="0";a:{try{var f=b.h.top.location.href}catch(ea){f=2;break a}f=f?f===b.i.location.href?0:1:2}e=(e.frm=f,e);try{e.u_tz=-(new Date).getTimezoneOffset();var g=void 0===g?Gh:g;try{var h=g.history.length}catch(ea){h=0}e.u_his=h;var k;e.u_h=null==(k=Gh.screen)?void 0:k.height;var m;e.u_w=null==(m=Gh.screen)?void 0:m.width;var q;e.u_ah=null==(q=Gh.screen)?void 0:q.availHeight;var r;e.u_aw=
null==(r=Gh.screen)?void 0:r.availWidth;var w;e.u_cd=null==(w=Gh.screen)?void 0:w.colorDepth}catch(ea){}h=b.h;try{var t=h.screenX;var y=h.screenY}catch(ea){}try{var E=h.outerWidth;var F=h.outerHeight}catch(ea){}try{var O=h.innerWidth;var N=h.innerHeight}catch(ea){}try{var R=h.screenLeft;var ca=h.screenTop}catch(ea){}try{O=h.innerWidth,N=h.innerHeight}catch(ea){}try{var U=h.screen.availWidth;var rb=h.screen.availTop}catch(ea){}t=[R,ca,t,y,U,rb,E,F,O,N];try{var Va=(b.h.top||window).document,na="CSS1Compat"==
Va.compatMode?Va.documentElement:Va.body;var I=(new lf(na.clientWidth,na.clientHeight)).round()}catch(ea){I=new lf(-12245933,-12245933)}Va=I;I={};var ma=void 0===ma?z:ma;na=new ai;ma.SVGElement&&ma.document.createElementNS&&na.set(0);y=Rh();y["allow-top-navigation-by-user-activation"]&&na.set(1);y["allow-popups-to-escape-sandbox"]&&na.set(2);ma.crypto&&ma.crypto.subtle&&na.set(3);ma.TextDecoder&&ma.TextEncoder&&na.set(4);ma=bi(na);I.bc=ma;I.bih=Va.height;I.biw=Va.width;I.brdim=t.join();b=b.i;b=(I.vis=
b.prerendering?3:{visible:1,hidden:2,prerender:3,preview:4,unloaded:5}[b.visibilityState||b.webkitVisibilityState||b.mozVisibilityState||""]||0,I.wgl=!!Gh.WebGLRenderingContext,I);c=d.call(c,e,b);c.ca_type="image";a&&(c.bid=a);return c}
var gl=new function(){var a=window.document;this.h=window;this.i=a};
A("yt.ads_.signals_.getAdSignalsString",function(a){return al(fl(a))});Date.now();var hl="XMLHttpRequest"in z?function(){return new XMLHttpRequest}:null;
function il(){if(!hl)return null;var a=hl();return"open"in a?a:null}
function jl(a){switch(a&&"status"in a?a.status:-1){case 200:case 201:case 202:case 203:case 204:case 205:case 206:case 304:return!0;default:return!1}}
;function kl(a,b){"function"===typeof a&&(a=Qk(a));return window.setTimeout(a,b)}
;var ll={Authorization:"AUTHORIZATION","X-Goog-EOM-Visitor-Id":"EOM_VISITOR_DATA","X-Goog-Visitor-Id":"SANDBOXED_VISITOR_ID","X-Youtube-Domain-Admin-State":"DOMAIN_ADMIN_STATE","X-Youtube-Chrome-Connected":"CHROME_CONNECTED_HEADER","X-YouTube-Client-Name":"INNERTUBE_CONTEXT_CLIENT_NAME","X-YouTube-Client-Version":"INNERTUBE_CONTEXT_CLIENT_VERSION","X-YouTube-Delegation-Context":"INNERTUBE_CONTEXT_SERIALIZED_DELEGATION_CONTEXT","X-YouTube-Device":"DEVICE","X-Youtube-Identity-Token":"ID_TOKEN","X-YouTube-Page-CL":"PAGE_CL",
"X-YouTube-Page-Label":"PAGE_BUILD_LABEL","X-YouTube-Variants-Checksum":"VARIANTS_CHECKSUM"},ml="app debugcss debugjs expflag force_ad_params force_ad_encrypted force_viral_ad_response_params forced_experiments innertube_snapshots innertube_goldens internalcountrycode internalipoverride absolute_experiments conditional_experiments sbb sr_bns_address".split(" ").concat(ia(Vh)),nl=!1;
function ol(a,b){b=void 0===b?{}:b;var c=el(a),d=M("web_ajax_ignore_global_headers_if_set"),e;for(e in ll){var f=L(ll[e]);"X-Goog-Visitor-Id"!==e||f||(f=L("VISITOR_DATA"));!f||!c&&nc(a)||d&&void 0!==b[e]||(b[e]=f)}"X-Goog-EOM-Visitor-Id"in b&&"X-Goog-Visitor-Id"in b&&delete b["X-Goog-Visitor-Id"];if(c||!nc(a))b["X-YouTube-Utc-Offset"]=String(-(new Date).getTimezoneOffset());if(c||!nc(a)){try{var g=(new Intl.DateTimeFormat).resolvedOptions().timeZone}catch(h){}g&&(b["X-YouTube-Time-Zone"]=g)}document.location.hostname.endsWith("youtubeeducation.com")||
!c&&nc(a)||(b["X-YouTube-Ad-Signals"]=al(fl()));return b}
function pl(a){var b=window.location.search,c=nc(a);M("debug_handle_relative_url_for_query_forward_killswitch")||!c&&el(a)&&(c=document.location.hostname);var d=lc(mc(5,a));d=(c=c&&(c.endsWith("youtube.com")||c.endsWith("youtube-nocookie.com")))&&d&&d.startsWith("/api/");if(!c||d)return a;var e=bl(b),f={};fb(ml,function(g){e[g]&&(f[g]=e[g])});
return dl(a,f||{},!1)}
function ql(a,b){var c=b.format||"JSON";a=rl(a,b);var d=sl(a,b),e=!1,f=tl(a,function(k){if(!e){e=!0;h&&window.clearTimeout(h);var m=jl(k),q=null,r=400<=k.status&&500>k.status,w=500<=k.status&&600>k.status;if(m||r||w)q=ul(a,c,k,b.convertToSafeHtml);if(m)a:if(k&&204==k.status)m=!0;else{switch(c){case "XML":m=0==parseInt(q&&q.return_code,10);break a;case "RAW":m=!0;break a}m=!!q}q=q||{};r=b.context||z;m?b.onSuccess&&b.onSuccess.call(r,k,q):b.onError&&b.onError.call(r,k,q);b.onFinish&&b.onFinish.call(r,
k,q)}},b.method,d,b.headers,b.responseType,b.withCredentials);
d=b.timeout||0;if(b.onTimeout&&0<d){var g=b.onTimeout;var h=kl(function(){e||(e=!0,f.abort(),window.clearTimeout(h),g.call(b.context||z,f))},d)}return f}
function rl(a,b){b.includeDomain&&(a=document.location.protocol+"//"+document.location.hostname+(document.location.port?":"+document.location.port:"")+a);var c=L("XSRF_FIELD_NAME");if(b=b.urlParams)b[c]&&delete b[c],a=dl(a,b||{},!0);return a}
function sl(a,b){var c=L("XSRF_FIELD_NAME"),d=L("XSRF_TOKEN"),e=b.postBody||"",f=b.postParams,g=L("XSRF_FIELD_NAME"),h;b.headers&&(h=b.headers["Content-Type"]);b.excludeXsrf||nc(a)&&!b.withCredentials&&nc(a)!=document.location.hostname||"POST"!=b.method||h&&"application/x-www-form-urlencoded"!=h||b.postParams&&b.postParams[g]||(f||(f={}),f[c]=d);(M("ajax_parse_query_data_only_when_filled")&&f&&0<Object.keys(f).length||f)&&"string"===typeof e&&(e=bl(e),xb(e,f),e=b.postBodyFormat&&"JSON"==b.postBodyFormat?
JSON.stringify(e):sc(e));f=e||f&&!pb(f);!nl&&f&&"POST"!=b.method&&(nl=!0,Rk(Error("AJAX request with postData should use POST")));return e}
function ul(a,b,c,d){var e=null;switch(b){case "JSON":try{var f=c.responseText}catch(g){throw d=Error("Error reading responseText"),d.params=a,Sk(d),g;}a=c.getResponseHeader("Content-Type")||"";f&&0<=a.indexOf("json")&&(")]}'\n"===f.substring(0,5)&&(f=f.substring(5)),e=JSON.parse(f));break;case "XML":if(a=(a=c.responseXML)?vl(a):null)e={},fb(a.getElementsByTagName("*"),function(g){e[g.tagName]=wl(g)})}d&&xl(e);
return e}
function xl(a){if(Qa(a))for(var b in a){var c;(c="html_content"==b)||(c=b.length-5,c=0<=c&&b.indexOf("_html",c)==c);if(c){c=b;Cb("HTML that is escaped and sanitized server-side and passed through yt.net.ajax");var d=a[b],e=zb();d=e?e.createHTML(d):d;a[c]=new dc(d)}else xl(a[b])}}
function vl(a){return a?(a=("responseXML"in a?a.responseXML:a).getElementsByTagName("root"))&&0<a.length?a[0]:null:null}
function wl(a){var b="";fb(a.childNodes,function(c){b+=c.nodeValue});
return b}
function tl(a,b,c,d,e,f,g){function h(){4==(k&&"readyState"in k?k.readyState:0)&&b&&Qk(b)(k)}
c=void 0===c?"GET":c;d=void 0===d?"":d;var k=il();if(!k)return null;"onloadend"in k?k.addEventListener("loadend",h,!1):k.onreadystatechange=h;M("debug_forward_web_query_parameters")&&(a=pl(a));k.open(c,a,!0);f&&(k.responseType=f);g&&(k.withCredentials=!0);c="POST"==c&&(void 0===window.FormData||!(d instanceof FormData));if(e=ol(a,e))for(var m in e)k.setRequestHeader(m,e[m]),"content-type"==m.toLowerCase()&&(c=!1);c&&k.setRequestHeader("Content-Type","application/x-www-form-urlencoded");k.send(d);
return k}
;function yl(a){var b=this;this.i=void 0;this.h=!1;a.addEventListener("beforeinstallprompt",function(c){c.preventDefault();b.i=c});
a.addEventListener("appinstalled",function(){b.h=!0},{once:!0})}
function zl(){if(!z.matchMedia)return"WEB_DISPLAY_MODE_UNKNOWN";try{return z.matchMedia("(display-mode: standalone)").matches?"WEB_DISPLAY_MODE_STANDALONE":z.matchMedia("(display-mode: minimal-ui)").matches?"WEB_DISPLAY_MODE_MINIMAL_UI":z.matchMedia("(display-mode: fullscreen)").matches?"WEB_DISPLAY_MODE_FULLSCREEN":z.matchMedia("(display-mode: browser)").matches?"WEB_DISPLAY_MODE_BROWSER":"WEB_DISPLAY_MODE_UNKNOWN"}catch(a){return"WEB_DISPLAY_MODE_UNKNOWN"}}
;function Al(a,b,c,d,e){tg.set(""+a,b,{ib:c,path:"/",domain:void 0===d?"youtube.com":d,secure:void 0===e?!1:e})}
function Bl(a,b,c){tg.remove(""+a,void 0===b?"/":b,void 0===c?"youtube.com":c)}
function Cl(){if(!tg.isEnabled())return!1;if(!tg.Oa())return!0;tg.set("TESTCOOKIESENABLED","1",{ib:60});if("1"!==tg.get("TESTCOOKIESENABLED"))return!1;tg.remove("TESTCOOKIESENABLED");return!0}
;var Dl=B("ytglobal.prefsUserPrefsPrefs_")||{};A("ytglobal.prefsUserPrefsPrefs_",Dl);function El(){this.h=L("ALT_PREF_COOKIE_NAME","PREF");this.i=L("ALT_PREF_COOKIE_DOMAIN","youtube.com");var a=tg.get(""+this.h,void 0);if(a){a=decodeURIComponent(a).split("&");for(var b=0;b<a.length;b++){var c=a[b].split("="),d=c[0];(c=c[1])&&(Dl[d]=c.toString())}}}
El.prototype.get=function(a,b){Fl(a);Gl(a);a=void 0!==Dl[a]?Dl[a].toString():null;return null!=a?a:b?b:""};
El.prototype.set=function(a,b){Fl(a);Gl(a);if(null==b)throw Error("ExpectedNotNull");Dl[a]=b.toString()};
function Hl(a){return!!((Il("f"+(Math.floor(a/31)+1))||0)&1<<a%31)}
El.prototype.remove=function(a){Fl(a);Gl(a);delete Dl[a]};
El.prototype.clear=function(){for(var a in Dl)delete Dl[a]};
function Gl(a){if(/^f([1-9][0-9]*)$/.test(a))throw Error("ExpectedRegexMatch: "+a);}
function Fl(a){if(!/^\w+$/.test(a))throw Error("ExpectedRegexMismatch: "+a);}
function Il(a){a=void 0!==Dl[a]?Dl[a].toString():null;return null!=a&&/^[A-Fa-f0-9]+$/.test(a)?parseInt(a,16):null}
Na(El);var Jl={bluetooth:"CONN_DISCO",cellular:"CONN_CELLULAR_UNKNOWN",ethernet:"CONN_WIFI",none:"CONN_NONE",wifi:"CONN_WIFI",wimax:"CONN_CELLULAR_4G",other:"CONN_UNKNOWN",unknown:"CONN_UNKNOWN","slow-2g":"CONN_CELLULAR_2G","2g":"CONN_CELLULAR_2G","3g":"CONN_CELLULAR_3G","4g":"CONN_CELLULAR_4G"},Kl={CONN_DEFAULT:0,CONN_UNKNOWN:1,CONN_NONE:2,CONN_WIFI:3,CONN_CELLULAR_2G:4,CONN_CELLULAR_3G:5,CONN_CELLULAR_4G:6,CONN_CELLULAR_UNKNOWN:7,CONN_DISCO:8,CONN_CELLULAR_5G:9,CONN_WIFI_METERED:10,CONN_CELLULAR_5G_SA:11,
CONN_CELLULAR_5G_NSA:12,CONN_INVALID:31},Ll={EFFECTIVE_CONNECTION_TYPE_UNKNOWN:0,EFFECTIVE_CONNECTION_TYPE_OFFLINE:1,EFFECTIVE_CONNECTION_TYPE_SLOW_2G:2,EFFECTIVE_CONNECTION_TYPE_2G:3,EFFECTIVE_CONNECTION_TYPE_3G:4,EFFECTIVE_CONNECTION_TYPE_4G:5},Ml={"slow-2g":"EFFECTIVE_CONNECTION_TYPE_SLOW_2G","2g":"EFFECTIVE_CONNECTION_TYPE_2G","3g":"EFFECTIVE_CONNECTION_TYPE_3G","4g":"EFFECTIVE_CONNECTION_TYPE_4G"};function Nl(){var a=z.navigator;return a?a.connection:void 0}
function Ol(){var a=Nl();if(a){var b=Jl[a.type||"unknown"]||"CONN_UNKNOWN";a=Jl[a.effectiveType||"unknown"]||"CONN_UNKNOWN";"CONN_CELLULAR_UNKNOWN"===b&&"CONN_UNKNOWN"!==a&&(b=a);if("CONN_UNKNOWN"!==b)return b;if("CONN_UNKNOWN"!==a)return a}}
function Pl(){var a=Nl();if(null!=a&&a.effectiveType)return Ml.hasOwnProperty(a.effectiveType)?Ml[a.effectiveType]:"EFFECTIVE_CONNECTION_TYPE_UNKNOWN"}
;function Ql(){}
function Rl(a,b){return Sl(a,0,b)}
Ql.prototype.R=function(a,b){return Sl(a,1,b)};
function Tl(a,b){Sl(a,2,b)}
function Ul(a){var b=B("yt.scheduler.instance.addImmediateJob");b?b(a):a()}
;function Vl(){Ql.apply(this,arguments)}
u(Vl,Ql);function Wl(){Vl.h||(Vl.h=new Vl);return Vl.h}
function Sl(a,b,c){void 0!==c&&Number.isNaN(Number(c))&&(c=void 0);var d=B("yt.scheduler.instance.addJob");return d?d(a,b,c):void 0===c?(a(),NaN):kl(a,c||0)}
Vl.prototype.ea=function(a){if(void 0===a||!Number.isNaN(Number(a))){var b=B("yt.scheduler.instance.cancelJob");b?b(a):window.clearTimeout(a)}};
Vl.prototype.start=function(){var a=B("yt.scheduler.instance.start");a&&a()};
Vl.prototype.pause=function(){var a=B("yt.scheduler.instance.pause");a&&a()};
var $h=Wl();function P(a){var b=Ja.apply(1,arguments);var c=Error.call(this,a);this.message=c.message;"stack"in c&&(this.stack=c.stack);this.args=[].concat(ia(b))}
u(P,Error);function Xl(){try{return Yl(),!0}catch(a){return!1}}
function Yl(a){if(void 0!==L("DATASYNC_ID"))return L("DATASYNC_ID");throw new P("Datasync ID not set",void 0===a?"unknown":a);}
;var Zl=Wc||Xc;function $l(a){var b=Sb();return b?0<=b.toLowerCase().indexOf(a):!1}
;function am(a){var b=new Gi;(b=b.isAvailable()?a?new Mi(b,a):b:null)||(a=new Hi(a||"UserDataSharedStore"),b=a.isAvailable()?a:null);this.h=(a=b)?new Ci(a):null;this.i=document.domain||window.location.hostname}
am.prototype.set=function(a,b,c,d){c=c||31104E3;this.remove(a);if(this.h)try{this.h.set(a,b,Date.now()+1E3*c);return}catch(f){}var e="";if(d)try{e=escape(Bg(b))}catch(f){return}else e=escape(b);Al(a,e,c,this.i)};
am.prototype.get=function(a,b){var c=void 0,d=!this.h;if(!d)try{c=this.h.get(a)}catch(e){d=!0}if(d&&(c=tg.get(""+a,void 0))&&(c=unescape(c),b))try{c=JSON.parse(c)}catch(e){this.remove(a),c=void 0}return c};
am.prototype.remove=function(a){this.h&&this.h.remove(a);Bl(a,"/",this.i)};var bm=function(){var a;return function(){a||(a=new am("ytidb"));return a}}();
function cm(){var a;return null==(a=bm())?void 0:a.get("LAST_RESULT_ENTRY_KEY",!0)}
;var dm=[],em,fm=!1;function gm(){var a={};for(em=new hm(void 0===a.handleError?im:a.handleError,void 0===a.logEvent?jm:a.logEvent);0<dm.length;)switch(a=dm.shift(),a.type){case "ERROR":em.handleError(a.payload);break;case "EVENT":em.logEvent(a.eventType,a.payload)}}
function km(a){fm||(em?em.handleError(a):(dm.push({type:"ERROR",payload:a}),10<dm.length&&dm.shift()))}
function lm(a,b){fm||(em?em.logEvent(a,b):(dm.push({type:"EVENT",eventType:a,payload:b}),10<dm.length&&dm.shift()))}
;function mm(a){if(0<=a.indexOf(":"))throw Error("Database name cannot contain ':'");}
function nm(a){return a.substr(0,a.indexOf(":"))||a}
;var om={},pm=(om.AUTH_INVALID="No user identifier specified.",om.EXPLICIT_ABORT="Transaction was explicitly aborted.",om.IDB_NOT_SUPPORTED="IndexedDB is not supported.",om.MISSING_INDEX="Index not created.",om.MISSING_OBJECT_STORES="Object stores not created.",om.DB_DELETED_BY_MISSING_OBJECT_STORES="Database is deleted because expected object stores were not created.",om.DB_REOPENED_BY_MISSING_OBJECT_STORES="Database is reopened because expected object stores were not created.",om.UNKNOWN_ABORT="Transaction was aborted for unknown reasons.",
om.QUOTA_EXCEEDED="The current transaction exceeded its quota limitations.",om.QUOTA_MAYBE_EXCEEDED="The current transaction may have failed because of exceeding quota limitations.",om.EXECUTE_TRANSACTION_ON_CLOSED_DB="Can't start a transaction on a closed database",om.INCOMPATIBLE_DB_VERSION="The binary is incompatible with the database version",om),qm={},rm=(qm.AUTH_INVALID="ERROR",qm.EXECUTE_TRANSACTION_ON_CLOSED_DB="WARNING",qm.EXPLICIT_ABORT="IGNORED",qm.IDB_NOT_SUPPORTED="ERROR",qm.MISSING_INDEX=
"WARNING",qm.MISSING_OBJECT_STORES="ERROR",qm.DB_DELETED_BY_MISSING_OBJECT_STORES="WARNING",qm.DB_REOPENED_BY_MISSING_OBJECT_STORES="WARNING",qm.QUOTA_EXCEEDED="WARNING",qm.QUOTA_MAYBE_EXCEEDED="WARNING",qm.UNKNOWN_ABORT="WARNING",qm.INCOMPATIBLE_DB_VERSION="WARNING",qm),sm={},tm=(sm.AUTH_INVALID=!1,sm.EXECUTE_TRANSACTION_ON_CLOSED_DB=!1,sm.EXPLICIT_ABORT=!1,sm.IDB_NOT_SUPPORTED=!1,sm.MISSING_INDEX=!1,sm.MISSING_OBJECT_STORES=!1,sm.DB_DELETED_BY_MISSING_OBJECT_STORES=!1,sm.DB_REOPENED_BY_MISSING_OBJECT_STORES=
!1,sm.QUOTA_EXCEEDED=!1,sm.QUOTA_MAYBE_EXCEEDED=!0,sm.UNKNOWN_ABORT=!0,sm.INCOMPATIBLE_DB_VERSION=!1,sm);function um(a,b,c,d,e){b=void 0===b?{}:b;c=void 0===c?pm[a]:c;d=void 0===d?rm[a]:d;e=void 0===e?tm[a]:e;P.call(this,c,Object.assign({},{name:"YtIdbKnownError",isSw:void 0===self.document,isIframe:self!==self.top,type:a},b));this.type=a;this.message=c;this.level=d;this.h=e;Object.setPrototypeOf(this,um.prototype)}
u(um,P);function vm(a,b){um.call(this,"MISSING_OBJECT_STORES",{expectedObjectStores:b,foundObjectStores:a},pm.MISSING_OBJECT_STORES);Object.setPrototypeOf(this,vm.prototype)}
u(vm,um);function wm(a,b){var c=Error.call(this);this.message=c.message;"stack"in c&&(this.stack=c.stack);this.index=a;this.objectStore=b;Object.setPrototypeOf(this,wm.prototype)}
u(wm,Error);var xm=["The database connection is closing","Can't start a transaction on a closed database","A mutation operation was attempted on a database that did not allow mutations"];
function ym(a,b,c,d){b=nm(b);var e=a instanceof Error?a:Error("Unexpected error: "+a);if(e instanceof um)return e;a={objectStoreNames:c,dbName:b,dbVersion:d};if("QuotaExceededError"===e.name)return new um("QUOTA_EXCEEDED",a);if(Yc&&"UnknownError"===e.name)return new um("QUOTA_MAYBE_EXCEEDED",a);if(e instanceof wm)return new um("MISSING_INDEX",Object.assign({},a,{objectStore:e.objectStore,index:e.index}));if("InvalidStateError"===e.name&&xm.some(function(f){return e.message.includes(f)}))return new um("EXECUTE_TRANSACTION_ON_CLOSED_DB",
a);
if("AbortError"===e.name)return new um("UNKNOWN_ABORT",a,e.message);e.args=[Object.assign({},a,{name:"IdbError",jc:e.name})];e.level="WARNING";return e}
function zm(a,b,c){var d=cm();return new um("IDB_NOT_SUPPORTED",{context:{caller:a,publicName:b,version:c,hasSucceededOnce:null==d?void 0:d.hasSucceededOnce}})}
;function Am(a){if(!a)throw Error();throw a;}
function Bm(a){return a}
function Cm(a){this.h=a}
function Dm(a){function b(e){if("PENDING"===d.state.status){d.state={status:"REJECTED",reason:e};e=p(d.i);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
function c(e){if("PENDING"===d.state.status){d.state={status:"FULFILLED",value:e};e=p(d.h);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
var d=this;this.state={status:"PENDING"};this.h=[];this.i=[];a=a.h;try{a(c,b)}catch(e){b(e)}}
Dm.all=function(a){return new Dm(new Cm(function(b,c){var d=[],e=a.length;0===e&&b(d);for(var f={za:0};f.za<a.length;f={za:f.za},++f.za)Dm.resolve(a[f.za]).then(function(g){return function(h){d[g.za]=h;e--;0===e&&b(d)}}(f)).catch(function(g){c(g)})}))};
Dm.resolve=function(a){return new Dm(new Cm(function(b,c){a instanceof Dm?a.then(b,c):b(a)}))};
Dm.reject=function(a){return new Dm(new Cm(function(b,c){c(a)}))};
Dm.prototype.then=function(a,b){var c=this,d=null!=a?a:Bm,e=null!=b?b:Am;return new Dm(new Cm(function(f,g){"PENDING"===c.state.status?(c.h.push(function(){Em(c,c,d,f,g)}),c.i.push(function(){Fm(c,c,e,f,g)})):"FULFILLED"===c.state.status?Em(c,c,d,f,g):"REJECTED"===c.state.status&&Fm(c,c,e,f,g)}))};
Dm.prototype.catch=function(a){return this.then(void 0,a)};
function Em(a,b,c,d,e){try{if("FULFILLED"!==a.state.status)throw Error("calling handleResolve before the promise is fulfilled.");var f=c(a.state.value);f instanceof Dm?Gm(a,b,f,d,e):d(f)}catch(g){e(g)}}
function Fm(a,b,c,d,e){try{if("REJECTED"!==a.state.status)throw Error("calling handleReject before the promise is rejected.");var f=c(a.state.reason);f instanceof Dm?Gm(a,b,f,d,e):d(f)}catch(g){e(g)}}
function Gm(a,b,c,d,e){b===c?e(new TypeError("Circular promise chain detected.")):c.then(function(f){f instanceof Dm?Gm(a,b,f,d,e):d(f)},function(f){e(f)})}
;function Hm(a,b,c){function d(){c(a.error);f()}
function e(){b(a.result);f()}
function f(){try{a.removeEventListener("success",e),a.removeEventListener("error",d)}catch(g){}}
a.addEventListener("success",e);a.addEventListener("error",d)}
function Im(a){return new Promise(function(b,c){Hm(a,b,c)})}
function Jm(a){return new Dm(new Cm(function(b,c){Hm(a,b,c)}))}
;function Km(a,b){return new Dm(new Cm(function(c,d){function e(){var f=a?b(a):null;f?f.then(function(g){a=g;e()},d):c()}
e()}))}
;var Lm=window,Q=Lm.ytcsi&&Lm.ytcsi.now?Lm.ytcsi.now:Lm.performance&&Lm.performance.timing&&Lm.performance.now&&Lm.performance.timing.navigationStart?function(){return Lm.performance.timing.navigationStart+Lm.performance.now()}:function(){return(new Date).getTime()};function Mm(a,b){this.h=a;this.options=b;this.transactionCount=0;this.j=Math.round(Q());this.i=!1}
l=Mm.prototype;l.add=function(a,b,c){return Nm(this,[a],{mode:"readwrite",T:!0},function(d){return d.objectStore(a).add(b,c)})};
l.clear=function(a){return Nm(this,[a],{mode:"readwrite",T:!0},function(b){return b.objectStore(a).clear()})};
l.close=function(){this.h.close();var a;(null==(a=this.options)?0:a.closed)&&this.options.closed()};
l.count=function(a,b){return Nm(this,[a],{mode:"readonly",T:!0},function(c){return c.objectStore(a).count(b)})};
function Om(a,b,c){a=a.h.createObjectStore(b,c);return new Pm(a)}
l.delete=function(a,b){return Nm(this,[a],{mode:"readwrite",T:!0},function(c){return c.objectStore(a).delete(b)})};
l.get=function(a,b){return Nm(this,[a],{mode:"readonly",T:!0},function(c){return c.objectStore(a).get(b)})};
function Qm(a,b,c){return Nm(a,[b],{mode:"readwrite",T:!0},function(d){d=d.objectStore(b);return Jm(d.h.put(c,void 0))})}
l.objectStoreNames=function(){return Array.from(this.h.objectStoreNames)};
function Nm(a,b,c,d){var e,f,g,h,k,m,q,r,w,t,y,E;return x(function(F){switch(F.h){case 1:var O={mode:"readonly",T:!1,tag:"IDB_TRANSACTION_TAG_UNKNOWN"};"string"===typeof c?O.mode=c:Object.assign(O,c);e=O;a.transactionCount++;f=e.T?3:1;g=0;case 2:if(h){F.u(3);break}g++;k=Math.round(Q());ya(F,4);m=a.h.transaction(b,e.mode);O=new Rm(m);O=Sm(O,d);return v(F,O,6);case 6:return q=F.i,r=Math.round(Q()),Tm(a,k,r,g,void 0,b.join(),e),F.return(q);case 4:w=Aa(F);t=Math.round(Q());y=ym(w,a.h.name,b.join(),a.h.version);
if((E=y instanceof um&&!y.h)||g>=f)Tm(a,k,t,g,y,b.join(),e),h=y;F.u(2);break;case 3:return F.return(Promise.reject(h))}})}
function Tm(a,b,c,d,e,f,g){b=c-b;e?(e instanceof um&&("QUOTA_EXCEEDED"===e.type||"QUOTA_MAYBE_EXCEEDED"===e.type)&&lm("QUOTA_EXCEEDED",{dbName:nm(a.h.name),objectStoreNames:f,transactionCount:a.transactionCount,transactionMode:g.mode}),e instanceof um&&"UNKNOWN_ABORT"===e.type&&(c-=a.j,0>c&&c>=Math.pow(2,31)&&(c=0),lm("TRANSACTION_UNEXPECTEDLY_ABORTED",{objectStoreNames:f,transactionDuration:b,transactionCount:a.transactionCount,dbDuration:c}),a.i=!0),Um(a,!1,d,f,b,g.tag),km(e)):Um(a,!0,d,f,b,g.tag)}
function Um(a,b,c,d,e,f){lm("TRANSACTION_ENDED",{objectStoreNames:d,connectionHasUnknownAbortedTransaction:a.i,duration:e,isSuccessful:b,tryCount:c,tag:void 0===f?"IDB_TRANSACTION_TAG_UNKNOWN":f})}
l.getName=function(){return this.h.name};
function Pm(a){this.h=a}
l=Pm.prototype;l.add=function(a,b){return Jm(this.h.add(a,b))};
l.autoIncrement=function(){return this.h.autoIncrement};
l.clear=function(){return Jm(this.h.clear()).then(function(){})};
function Vm(a,b,c){a.h.createIndex(b,c,{unique:!1})}
l.count=function(a){return Jm(this.h.count(a))};
function Wm(a,b){return Xm(a,{query:b},function(c){return c.delete().then(function(){return c.continue()})}).then(function(){})}
l.delete=function(a){return a instanceof IDBKeyRange?Wm(this,a):Jm(this.h.delete(a))};
l.get=function(a){return Jm(this.h.get(a))};
l.index=function(a){try{return new Ym(this.h.index(a))}catch(b){if(b instanceof Error&&"NotFoundError"===b.name)throw new wm(a,this.h.name);throw b;}};
l.getName=function(){return this.h.name};
l.keyPath=function(){return this.h.keyPath};
function Xm(a,b,c){a=a.h.openCursor(b.query,b.direction);return Zm(a).then(function(d){return Km(d,c)})}
function Rm(a){var b=this;this.h=a;this.j=new Map;this.i=!1;this.done=new Promise(function(c,d){b.h.addEventListener("complete",function(){c()});
b.h.addEventListener("error",function(e){e.currentTarget===e.target&&d(b.h.error)});
b.h.addEventListener("abort",function(){var e=b.h.error;if(e)d(e);else if(!b.i){e=um;for(var f=b.h.objectStoreNames,g=[],h=0;h<f.length;h++){var k=f.item(h);if(null===k)throw Error("Invariant: item in DOMStringList is null");g.push(k)}e=new e("UNKNOWN_ABORT",{objectStoreNames:g.join(),dbName:b.h.db.name,mode:b.h.mode});d(e)}})})}
function Sm(a,b){var c=new Promise(function(d,e){try{b(a).then(function(f){d(f)}).catch(e)}catch(f){e(f),a.abort()}});
return Promise.all([c,a.done]).then(function(d){return p(d).next().value})}
Rm.prototype.abort=function(){this.h.abort();this.i=!0;throw new um("EXPLICIT_ABORT");};
Rm.prototype.objectStore=function(a){a=this.h.objectStore(a);var b=this.j.get(a);b||(b=new Pm(a),this.j.set(a,b));return b};
function Ym(a){this.h=a}
l=Ym.prototype;l.count=function(a){return Jm(this.h.count(a))};
l.delete=function(a){return $m(this,{query:a},function(b){return b.delete().then(function(){return b.continue()})})};
l.get=function(a){return Jm(this.h.get(a))};
l.getKey=function(a){return Jm(this.h.getKey(a))};
l.keyPath=function(){return this.h.keyPath};
l.unique=function(){return this.h.unique};
function $m(a,b,c){a=a.h.openCursor(void 0===b.query?null:b.query,void 0===b.direction?"next":b.direction);return Zm(a).then(function(d){return Km(d,c)})}
function an(a,b){this.request=a;this.cursor=b}
function Zm(a){return Jm(a).then(function(b){return b?new an(a,b):null})}
l=an.prototype;l.advance=function(a){this.cursor.advance(a);return Zm(this.request)};
l.continue=function(a){this.cursor.continue(a);return Zm(this.request)};
l.delete=function(){return Jm(this.cursor.delete()).then(function(){})};
l.getKey=function(){return this.cursor.key};
l.getValue=function(){return this.cursor.value};
l.update=function(a){return Jm(this.cursor.update(a))};function bn(a,b,c){return new Promise(function(d,e){function f(){w||(w=new Mm(g.result,{closed:r}));return w}
var g=void 0!==b?self.indexedDB.open(a,b):self.indexedDB.open(a);var h=c.Hc,k=c.blocking,m=c.xd,q=c.upgrade,r=c.closed,w;g.addEventListener("upgradeneeded",function(t){try{if(null===t.newVersion)throw Error("Invariant: newVersion on IDbVersionChangeEvent is null");if(null===g.transaction)throw Error("Invariant: transaction on IDbOpenDbRequest is null");t.dataLoss&&"none"!==t.dataLoss&&lm("IDB_DATA_CORRUPTED",{reason:t.dataLossMessage||"unknown reason",dbName:nm(a)});var y=f(),E=new Rm(g.transaction);
q&&q(y,function(F){return t.oldVersion<F&&t.newVersion>=F},E);
E.done.catch(function(F){e(F)})}catch(F){e(F)}});
g.addEventListener("success",function(){var t=g.result;k&&t.addEventListener("versionchange",function(){k(f())});
t.addEventListener("close",function(){lm("IDB_UNEXPECTEDLY_CLOSED",{dbName:nm(a),dbVersion:t.version});m&&m()});
d(f())});
g.addEventListener("error",function(){e(g.error)});
h&&g.addEventListener("blocked",function(){h()})})}
function cn(a,b,c){c=void 0===c?{}:c;return bn(a,b,c)}
function dn(a,b){b=void 0===b?{}:b;var c,d,e,f;return x(function(g){if(1==g.h)return ya(g,2),c=self.indexedDB.deleteDatabase(a),d=b,(e=d.Hc)&&c.addEventListener("blocked",function(){e()}),v(g,Im(c),4);
if(2!=g.h)return za(g,0);f=Aa(g);throw ym(f,a,"",-1);})}
;function en(a){return new Promise(function(b){Tl(function(){b()},a)})}
function fn(a,b){this.name=a;this.options=b;this.l=!0;this.o=this.m=0;this.i=500}
fn.prototype.j=function(a,b,c){c=void 0===c?{}:c;return cn(a,b,c)};
fn.prototype.delete=function(a){a=void 0===a?{}:a;return dn(this.name,a)};
function gn(a,b){return new um("INCOMPATIBLE_DB_VERSION",{dbName:a.name,oldVersion:a.options.version,newVersion:b})}
function hn(a,b){if(!b)throw zm("openWithToken",nm(a.name));return jn(a)}
function jn(a){function b(){var f,g,h,k,m,q,r,w,t,y;return x(function(E){switch(E.h){case 1:return g=null!=(f=Error().stack)?f:"",ya(E,2),v(E,a.j(a.name,a.options.version,d),4);case 4:h=E.i;for(var F=a.options,O=[],N=p(Object.keys(F.Da)),R=N.next();!R.done;R=N.next()){R=R.value;var ca=F.Da[R],U=void 0===ca.hd?Number.MAX_VALUE:ca.hd;!(h.h.version>=ca.Ia)||h.h.version>=U||h.h.objectStoreNames.contains(R)||O.push(R)}k=O;if(0===k.length){E.u(5);break}m=Object.keys(a.options.Da);q=h.objectStoreNames();
if(a.o<Lk("ytidb_reopen_db_retries",0))return a.o++,h.close(),km(new um("DB_REOPENED_BY_MISSING_OBJECT_STORES",{dbName:a.name,expectedObjectStores:m,foundObjectStores:q})),E.return(b());if(!(a.m<Lk("ytidb_remake_db_retries",1))){E.u(6);break}a.m++;if(!M("ytidb_remake_db_enable_backoff_delay")){E.u(7);break}return v(E,en(a.i),8);case 8:a.i*=2;case 7:return v(E,a.delete(),9);case 9:return km(new um("DB_DELETED_BY_MISSING_OBJECT_STORES",{dbName:a.name,expectedObjectStores:m,foundObjectStores:q})),E.return(b());
case 6:throw new vm(q,m);case 5:return E.return(h);case 2:r=Aa(E);if(r instanceof DOMException?"VersionError"!==r.name:"DOMError"in self&&r instanceof DOMError?"VersionError"!==r.name:!(r instanceof Object&&"message"in r)||"An attempt was made to open a database using a lower version than the existing version."!==r.message){E.u(10);break}return v(E,a.j(a.name,void 0,Object.assign({},d,{upgrade:void 0})),11);case 11:w=E.i;t=w.h.version;if(void 0!==a.options.version&&t>a.options.version+1)throw w.close(),
a.l=!1,gn(a,t);return E.return(w);case 10:throw c(),r instanceof Error&&!M("ytidb_async_stack_killswitch")&&(r.stack=r.stack+"\n"+g.substring(g.indexOf("\n")+1)),ym(r,a.name,"",null!=(y=a.options.version)?y:-1);}})}
function c(){a.h===e&&(a.h=void 0)}
if(!a.l)throw gn(a);if(a.h)return a.h;var d={blocking:function(f){f.close()},
closed:c,xd:c,upgrade:a.options.upgrade};var e=b();a.h=e;return a.h}
;var kn=new fn("YtIdbMeta",{Da:{databases:{Ia:1}},upgrade:function(a,b){b(1)&&Om(a,"databases",{keyPath:"actualName"})}});
function ln(a,b){var c;return x(function(d){if(1==d.h)return v(d,hn(kn,b),2);c=d.i;return d.return(Nm(c,["databases"],{T:!0,mode:"readwrite"},function(e){var f=e.objectStore("databases");return f.get(a.actualName).then(function(g){if(g?a.actualName!==g.actualName||a.publicName!==g.publicName||a.userIdentifier!==g.userIdentifier:1)return Jm(f.h.put(a,void 0)).then(function(){})})}))})}
function mn(a,b){var c;return x(function(d){if(1==d.h)return a?v(d,hn(kn,b),2):d.return();c=d.i;return d.return(c.delete("databases",a))})}
function nn(a,b){var c,d;return x(function(e){return 1==e.h?(c=[],v(e,hn(kn,b),2)):3!=e.h?(d=e.i,v(e,Nm(d,["databases"],{T:!0,mode:"readonly"},function(f){c.length=0;return Xm(f.objectStore("databases"),{},function(g){a(g.getValue())&&c.push(g.getValue());return g.continue()})}),3)):e.return(c)})}
function on(a){return nn(function(b){return"LogsDatabaseV2"===b.publicName&&void 0!==b.userIdentifier},a)}
function pn(a,b,c){return nn(function(d){return c?void 0!==d.userIdentifier&&!a.includes(d.userIdentifier)&&c.includes(d.publicName):void 0!==d.userIdentifier&&!a.includes(d.userIdentifier)},b)}
function qn(a){var b,c;return x(function(d){if(1==d.h)return b=Yl("YtIdbMeta hasAnyMeta other"),v(d,nn(function(e){return void 0!==e.userIdentifier&&e.userIdentifier!==b},a),2);
c=d.i;return d.return(0<c.length)})}
;var rn,sn=new function(){}(new function(){});
function tn(){var a,b,c,d;return x(function(e){switch(e.h){case 1:a=cm();if(null==(b=a)?0:b.hasSucceededOnce)return e.return(!0);var f;if(f=Zl)f=/WebKit\/([0-9]+)/.exec(Sb()),f=!!(f&&600<=parseInt(f[1],10));f&&(f=/WebKit\/([0-9]+)/.exec(Sb()),f=!(f&&602<=parseInt(f[1],10)));if(f||Jc)return e.return(!1);try{if(c=self,!(c.indexedDB&&c.IDBIndex&&c.IDBKeyRange&&c.IDBObjectStore))return e.return(!1)}catch(g){return e.return(!1)}if(!("IDBTransaction"in self&&"objectStoreNames"in IDBTransaction.prototype))return e.return(!1);
ya(e,2);d={actualName:"yt-idb-test-do-not-use",publicName:"yt-idb-test-do-not-use",userIdentifier:void 0};return v(e,ln(d,sn),4);case 4:return v(e,mn("yt-idb-test-do-not-use",sn),5);case 5:return e.return(!0);case 2:return Aa(e),e.return(!1)}})}
function un(){if(void 0!==rn)return rn;fm=!0;return rn=tn().then(function(a){fm=!1;var b;if(null!=(b=bm())&&b.h){var c;b={hasSucceededOnce:(null==(c=cm())?void 0:c.hasSucceededOnce)||a};var d;null==(d=bm())||d.set("LAST_RESULT_ENTRY_KEY",b,2592E3,!0)}return a})}
function vn(){return B("ytglobal.idbToken_")||void 0}
function wn(){var a=vn();return a?Promise.resolve(a):un().then(function(b){(b=b?sn:void 0)&&A("ytglobal.idbToken_",b);return b})}
;var xn=0;function yn(a,b){xn||(xn=$h.R(function(){var c,d,e,f,g;return x(function(h){switch(h.h){case 1:return v(h,wn(),2);case 2:c=h.i;if(!c)return h.return();d=!0;ya(h,3);return v(h,pn(a,c,b),5);case 5:e=h.i;if(!e.length){d=!1;h.u(6);break}f=e[0];return v(h,dn(f.actualName),7);case 7:return v(h,mn(f.actualName,c),6);case 6:za(h,4);break;case 3:g=Aa(h),km(g),d=!1;case 4:$h.ea(xn),xn=0,d&&yn(a,b),h.h=0}})}))}
function zn(){var a;return x(function(b){return 1==b.h?v(b,wn(),2):(a=b.i)?b.return(qn(a)):b.return(!1)})}
new Eh;function An(a){if(!Xl())throw a=new um("AUTH_INVALID",{dbName:a}),km(a),a;var b=Yl();return{actualName:a+":"+b,publicName:a,userIdentifier:b}}
function Bn(a,b,c,d){var e,f,g,h,k,m;return x(function(q){switch(q.h){case 1:return f=null!=(e=Error().stack)?e:"",v(q,wn(),2);case 2:g=q.i;if(!g)throw h=zm("openDbImpl",a,b),M("ytidb_async_stack_killswitch")||(h.stack=h.stack+"\n"+f.substring(f.indexOf("\n")+1)),km(h),h;mm(a);k=c?{actualName:a,publicName:a,userIdentifier:void 0}:An(a);ya(q,3);return v(q,ln(k,g),5);case 5:return v(q,cn(k.actualName,b,d),6);case 6:return q.return(q.i);case 3:return m=Aa(q),ya(q,7),v(q,mn(k.actualName,g),9);case 9:za(q,
8);break;case 7:Aa(q);case 8:throw m;}})}
function Cn(a,b,c){c=void 0===c?{}:c;return Bn(a,b,!1,c)}
function Dn(a,b,c){c=void 0===c?{}:c;return Bn(a,b,!0,c)}
function En(a,b){b=void 0===b?{}:b;var c,d;return x(function(e){if(1==e.h)return v(e,wn(),2);if(3!=e.h){c=e.i;if(!c)return e.return();mm(a);d=An(a);return v(e,dn(d.actualName,b),3)}return v(e,mn(d.actualName,c),0)})}
function Fn(a,b,c){a=a.map(function(d){return x(function(e){return 1==e.h?v(e,dn(d.actualName,b),2):v(e,mn(d.actualName,c),0)})});
return Promise.all(a).then(function(){})}
function Gn(){var a=void 0===a?{}:a;var b,c;return x(function(d){if(1==d.h)return v(d,wn(),2);if(3!=d.h){b=d.i;if(!b)return d.return();mm("LogsDatabaseV2");return v(d,on(b),3)}c=d.i;return v(d,Fn(c,a,b),0)})}
function Hn(a,b){b=void 0===b?{}:b;var c;return x(function(d){if(1==d.h)return v(d,wn(),2);if(3!=d.h){c=d.i;if(!c)return d.return();mm(a);return v(d,dn(a,b),3)}return v(d,mn(a,c),0)})}
;function In(a,b){fn.call(this,a,b);this.options=b;mm(a)}
u(In,fn);function Jn(a,b){var c;return function(){c||(c=new In(a,b));return c}}
In.prototype.j=function(a,b,c){c=void 0===c?{}:c;return(this.options.pb?Dn:Cn)(a,b,Object.assign({},c))};
In.prototype.delete=function(a){a=void 0===a?{}:a;return(this.options.pb?Hn:En)(this.name,a)};
function Kn(a,b){return Jn(a,b)}
;var Ln={},Mn=Kn("ytGcfConfig",{Da:(Ln.coldConfigStore={Ia:1},Ln.hotConfigStore={Ia:1},Ln),pb:!1,upgrade:function(a,b){b(1)&&(Vm(Om(a,"hotConfigStore",{keyPath:"key",autoIncrement:!0}),"hotTimestampIndex","timestamp"),Vm(Om(a,"coldConfigStore",{keyPath:"key",autoIncrement:!0}),"coldTimestampIndex","timestamp"))},
version:1});function Nn(a){return hn(Mn(),a)}
function On(a,b,c){var d,e,f;return x(function(g){switch(g.h){case 1:return d={config:a,hashData:b,timestamp:Q()},v(g,Nn(c),2);case 2:return e=g.i,v(g,e.clear("hotConfigStore"),3);case 3:return v(g,Qm(e,"hotConfigStore",d),4);case 4:return f=g.i,g.return(f)}})}
function Pn(a,b,c,d){var e,f,g;return x(function(h){switch(h.h){case 1:return e={config:a,hashData:b,configData:c,timestamp:Q()},v(h,Nn(d),2);case 2:return f=h.i,v(h,f.clear("coldConfigStore"),3);case 3:return v(h,Qm(f,"coldConfigStore",e),4);case 4:return g=h.i,h.return(g)}})}
;function Qn(){}
function Rn(a,b,c){var d,e;return x(function(f){if(!M("gcf_config_store_update_enabled"))return f.u(0);a.h=b;A("yt.gcf.config.hotConfigGroup",a.h);a.hotHashData=c;A("yt.gcf.config.hotHashData",a.hotHashData);d=vn();return d?v(f,On(b,c,d),0):(e=zm("updateHotConfig token error"),Sk(e),f.u(0))})}
function Sn(a,b,c){var d,e;return x(function(f){if(!M("gcf_config_store_update_enabled"))return f.u(0);a.coldHashData=c;A("yt.gcf.config.coldHashData",a.coldHashData);d=vn();return d?b.configData?v(f,Pn(b,c,b.configData,d),0):f.u(0):(e=zm("updateColdConfig token error"),Sk(e),f.u(0))})}
;function Tn(){return"INNERTUBE_API_KEY"in Hk&&"INNERTUBE_API_VERSION"in Hk}
function Un(){return{innertubeApiKey:L("INNERTUBE_API_KEY"),innertubeApiVersion:L("INNERTUBE_API_VERSION"),yb:L("INNERTUBE_CONTEXT_CLIENT_CONFIG_INFO"),cc:L("INNERTUBE_CONTEXT_CLIENT_NAME","WEB"),Uc:L("INNERTUBE_CONTEXT_CLIENT_NAME",1),innertubeContextClientVersion:L("INNERTUBE_CONTEXT_CLIENT_VERSION"),ec:L("INNERTUBE_CONTEXT_HL"),dc:L("INNERTUBE_CONTEXT_GL"),Vc:L("INNERTUBE_HOST_OVERRIDE")||"",Xc:!!L("INNERTUBE_USE_THIRD_PARTY_AUTH",!1),Wc:!!L("INNERTUBE_OMIT_API_KEY_WHEN_AUTH_HEADER_IS_PRESENT",
!1),appInstallData:L("SERIALIZED_CLIENT_CONFIG_DATA")}}
function Vn(a){var b={client:{hl:a.ec,gl:a.dc,clientName:a.cc,clientVersion:a.innertubeContextClientVersion,configInfo:a.yb}};navigator.userAgent&&(b.client.userAgent=String(navigator.userAgent));var c=z.devicePixelRatio;c&&1!=c&&(b.client.screenDensityFloat=String(c));c=Mk();""!==c&&(b.client.experimentsToken=c);c=Nk();0<c.length&&(b.request={internalExperimentFlags:c});Wn(a,void 0,b);M("enable_third_party_info")&&Xn(void 0,b);Yn(void 0,b);Zn(a,void 0,b);$n(void 0,b);M("start_sending_config_hash")&&
ao(void 0,b);L("DELEGATED_SESSION_ID")&&!M("pageid_as_header_web")&&(b.user={onBehalfOfUser:L("DELEGATED_SESSION_ID")});a=Object;c=a.assign;for(var d=b.client,e={},f=p(Object.entries(bl(L("DEVICE","")))),g=f.next();!g.done;g=f.next()){var h=p(g.value);g=h.next().value;h=h.next().value;"cbrand"===g?e.deviceMake=h:"cmodel"===g?e.deviceModel=h:"cbr"===g?e.browserName=h:"cbrver"===g?e.browserVersion=h:"cos"===g?e.osName=h:"cosver"===g?e.osVersion=h:"cplatform"===g&&(e.platform=h)}b.client=c.call(a,d,
e);return b}
function bo(a){var b=new fj,c=new Xi;D(c,1,a.ec);D(c,2,a.dc);D(c,16,a.Uc);D(c,17,a.innertubeContextClientVersion);if(a.yb){var d=a.yb,e=new Ti;d.coldConfigData&&D(e,1,d.coldConfigData);d.appInstallData&&D(e,6,d.appInstallData);d.coldHashData&&D(e,3,d.coldHashData);d.hotHashData&&D(e,5,d.hotHashData);G(c,Ti,62,e)}(d=z.devicePixelRatio)&&1!=d&&D(c,65,d);d=Mk();""!==d&&D(c,54,d);d=Nk();if(0<d.length){e=new Zi;for(var f=0;f<d.length;f++){var g=new Ri;D(g,1,d[f].key);Vd(g,2,Si,d[f].value);ce(e,15,Ri,g)}G(b,
Zi,5,e)}Wn(a,c);M("enable_third_party_info")&&Xn(b);Yn(c);Zn(a,c);$n(c);M("start_sending_config_hash")&&ao(c);L("DELEGATED_SESSION_ID")&&!M("pageid_as_header_web")&&(a=new dj,D(a,3,L("DELEGATED_SESSION_ID")));a=p(Object.entries(bl(L("DEVICE",""))));for(d=a.next();!d.done;d=a.next())e=p(d.value),d=e.next().value,e=e.next().value,"cbrand"===d?D(c,12,e):"cmodel"===d?D(c,13,e):"cbr"===d?D(c,87,e):"cbrver"===d?D(c,88,e):"cos"===d?D(c,18,e):"cosver"===d?D(c,19,e):"cplatform"===d&&D(c,42,e);b.i(c);return b}
function Wn(a,b,c){a=a.cc;if("WEB"===a||"MWEB"===a||1===a||2===a)if(b){c=Xd(b,Ui,96)||new Ui;var d=zl();d=Object.keys(mj).indexOf(d);d=-1===d?null:d;null!==d&&D(c,3,d);G(b,Ui,96,c)}else c&&(c.client.mainAppWebInfo=null!=(d=c.client.mainAppWebInfo)?d:{},c.client.mainAppWebInfo.webDisplayMode=zl())}
function Xn(a,b){var c=B("yt.embedded_player.embed_url");c&&(a?(b=Xd(a,aj,7)||new aj,D(b,4,c),G(a,aj,7,b)):b&&(b.thirdParty={embedUrl:c}))}
function Yn(a,b){var c;if(M("web_log_memory_total_kbytes")&&(null==(c=z.navigator)?0:c.deviceMemory)){var d;c=null==(d=z.navigator)?void 0:d.deviceMemory;a?D(a,95,1E6*c):b&&(b.client.memoryTotalKbytes=""+1E6*c)}}
function Zn(a,b,c){if(a.appInstallData)if(b){var d;c=null!=(d=Xd(b,Ti,62))?d:new Ti;D(c,6,a.appInstallData);G(b,Ti,62,c)}else c&&(c.client.configInfo=c.client.configInfo||{},c.client.configInfo.appInstallData=a.appInstallData)}
function $n(a,b){var c=Ol();c&&(a?D(a,61,Kl[c]):b&&(b.client.connectionType=c));M("web_log_effective_connection_type")&&(c=Pl())&&(a?D(a,94,Ll[c]):b&&(b.client.effectiveConnectionType=c))}
function co(a,b,c){c=void 0===c?{}:c;var d={};L("EOM_VISITOR_DATA")?d={"X-Goog-EOM-Visitor-Id":L("EOM_VISITOR_DATA")}:d={"X-Goog-Visitor-Id":c.visitorData||L("VISITOR_DATA","")};if(b&&b.includes("www.youtube-nocookie.com"))return d;(b=c.kr||L("AUTHORIZATION"))||(a?b="Bearer "+B("gapi.auth.getToken")().jr:b=xg([]));b&&(d.Authorization=b,d["X-Goog-AuthUser"]=L("SESSION_INDEX",0),M("pageid_as_header_web")&&(d["X-Goog-PageId"]=L("DELEGATED_SESSION_ID")));return d}
function ao(a,b){Qn.h||(Qn.h=new Qn);var c=B("yt.gcf.config.coldConfigData");var d=B("yt.gcf.config.hotHashData");var e=B("yt.gcf.config.coldHashData");if(c&&e&&d)if(a){var f;b=null!=(f=Xd(a,Ti,62))?f:new Ti;D(b,1,c);D(b,3,e);D(b,5,d);G(a,Ti,62,b)}else b&&(b.client.configInfo=b.client.configInfo||{},b.client.configInfo.coldConfigData=c,b.client.configInfo.coldHashData=e,b.client.configInfo.hotHashData=d)}
;function eo(a){a=Object.assign({},a);delete a.Authorization;var b=xg();if(b){var c=new fi;c.update(L("INNERTUBE_API_KEY"));c.update(b);a.hash=ad(c.digest(),3)}return a}
;var fo;function go(){fo||(fo=new am("yt.innertube"));return fo}
function ho(a,b,c,d){if(d)return null;d=go().get("nextId",!0)||1;var e=go().get("requests",!0)||{};e[d]={method:a,request:b,authState:eo(c),requestTime:Math.round(Q())};go().set("nextId",d+1,86400,!0);go().set("requests",e,86400,!0);return d}
function io(a){var b=go().get("requests",!0)||{};delete b[a];go().set("requests",b,86400,!0)}
function jo(a){var b=go().get("requests",!0);if(b){for(var c in b){var d=b[c];if(!(6E4>Math.round(Q())-d.requestTime)){var e=d.authState,f=eo(co(!1));tb(e,f)&&(e=d.request,"requestTimeMs"in e&&(e.requestTimeMs=Math.round(Q())),ko(a,d.method,e,{}));delete b[c]}}go().set("requests",b,86400,!0)}}
;function lo(a){this.bb=this.h=!1;this.potentialEsfErrorCounter=this.i=0;this.handleError=function(){};
this.Ca=function(){};
this.now=Date.now;this.Ka=!1;var b;this.uc=null!=(b=a.uc)?b:100;var c;this.oc=null!=(c=a.oc)?c:1;var d;this.mc=null!=(d=a.mc)?d:2592E6;var e;this.kc=null!=(e=a.kc)?e:12E4;var f;this.nc=null!=(f=a.nc)?f:5E3;var g;this.H=null!=(g=a.H)?g:void 0;this.gb=!!a.gb;var h;this.fb=null!=(h=a.fb)?h:.1;var k;this.lb=null!=(k=a.lb)?k:10;a.handleError&&(this.handleError=a.handleError);a.Ca&&(this.Ca=a.Ca);a.Ka&&(this.Ka=a.Ka);a.bb&&(this.bb=a.bb);this.J=a.J;this.Y=a.Y;this.N=a.N;this.L=a.L;this.ia=a.ia;this.Db=
a.Db;this.Cb=a.Cb;mo(this)&&(!this.J||this.J("networkless_logging"))&&no(this)}
function no(a){mo(a)&&!a.Ka&&(a.h=!0,a.gb&&Math.random()<=a.fb&&a.N.Jc(a.H),oo(a),a.L.U()&&a.Ra(),a.L.ha(a.Db,a.Ra.bind(a)),a.L.ha(a.Cb,a.Qb.bind(a)))}
l=lo.prototype;l.writeThenSend=function(a,b){var c=this;b=void 0===b?{}:b;if(mo(this)&&this.h){var d={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0};this.N.set(d,this.H).then(function(e){d.id=e;c.L.U()&&po(c,d)}).catch(function(e){po(c,d);
qo(c,e)})}else this.ia(a,b)};
l.sendThenWrite=function(a,b,c){var d=this;b=void 0===b?{}:b;if(mo(this)&&this.h){var e={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0};this.J&&this.J("nwl_skip_retry")&&(e.skipRetry=c);if(this.L.U()||this.J&&this.J("nwl_aggressive_send_then_write")&&!e.skipRetry){if(!e.skipRetry){var f=b.onError?b.onError:function(){};
b.onError=function(g,h){return x(function(k){if(1==k.h)return v(k,d.N.set(e,d.H).catch(function(m){qo(d,m)}),2);
f(g,h);k.h=0})}}this.ia(a,b,e.skipRetry)}else this.N.set(e,this.H).catch(function(g){d.ia(a,b,e.skipRetry);
qo(d,g)})}else this.ia(a,b,this.J&&this.J("nwl_skip_retry")&&c)};
l.sendAndWrite=function(a,b){var c=this;b=void 0===b?{}:b;if(mo(this)&&this.h){var d={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0},e=!1,f=b.onSuccess?b.onSuccess:function(){};
d.options.onSuccess=function(g,h){void 0!==d.id?c.N.Ba(d.id,c.H):e=!0;c.L.va&&c.J&&c.J("vss_network_hint")&&c.L.va(!0);f(g,h)};
this.ia(d.url,d.options);this.N.set(d,this.H).then(function(g){d.id=g;e&&c.N.Ba(d.id,c.H)}).catch(function(g){qo(c,g)})}else this.ia(a,b)};
l.Ra=function(){var a=this;if(!mo(this))throw zm("throttleSend");this.i||(this.i=this.Y.R(function(){var b;return x(function(c){if(1==c.h)return v(c,a.N.ac("NEW",a.H),2);if(3!=c.h)return b=c.i,b?v(c,po(a,b),3):(a.Qb(),c.return());a.i&&(a.i=0,a.Ra());c.h=0})},this.uc))};
l.Qb=function(){this.Y.ea(this.i);this.i=0};
function po(a,b){var c,d;return x(function(e){switch(e.h){case 1:if(!mo(a))throw c=zm("immediateSend"),c;if(void 0===b.id){e.u(2);break}return v(e,a.N.Zc(b.id,a.H),3);case 3:(d=e.i)?b=d:a.Ca(Error("The request cannot be found in the database."));case 2:if(ro(a,b,a.mc)){e.u(4);break}a.Ca(Error("Networkless Logging: Stored logs request expired age limit"));if(void 0===b.id){e.u(5);break}return v(e,a.N.Ba(b.id,a.H),5);case 5:return e.return();case 4:b.skipRetry||(b=so(a,b));if(!b){e.u(0);break}if(!b.skipRetry||
void 0===b.id){e.u(8);break}return v(e,a.N.Ba(b.id,a.H),8);case 8:a.ia(b.url,b.options,!!b.skipRetry),e.h=0}})}
function so(a,b){if(!mo(a))throw zm("updateRequestHandlers");var c=b.options.onError?b.options.onError:function(){};
b.options.onError=function(e,f){var g,h,k;return x(function(m){switch(m.h){case 1:g=to(f);if(!(a.J&&a.J("nwl_consider_error_code")&&g||a.J&&!a.J("nwl_consider_error_code")&&a.potentialEsfErrorCounter<=a.lb)){m.u(2);break}if(!a.L.ob){m.u(3);break}return v(m,a.L.ob(),3);case 3:if(a.L.U()){m.u(2);break}c(e,f);if(!a.J||!a.J("nwl_consider_error_code")||void 0===(null==(h=b)?void 0:h.id)){m.u(6);break}return v(m,a.N.Hb(b.id,a.H,!1),6);case 6:return m.return();case 2:if(a.J&&a.J("nwl_consider_error_code")&&
!g&&a.potentialEsfErrorCounter>a.lb)return m.return();a.potentialEsfErrorCounter++;if(void 0===(null==(k=b)?void 0:k.id)){m.u(8);break}return b.sendCount<a.oc?v(m,a.N.Hb(b.id,a.H),12):v(m,a.N.Ba(b.id,a.H),8);case 12:a.Y.R(function(){a.L.U()&&a.Ra()},a.nc);
case 8:c(e,f),m.h=0}})};
var d=b.options.onSuccess?b.options.onSuccess:function(){};
b.options.onSuccess=function(e,f){var g;return x(function(h){if(1==h.h)return void 0===(null==(g=b)?void 0:g.id)?h.u(2):v(h,a.N.Ba(b.id,a.H),2);a.L.va&&a.J&&a.J("vss_network_hint")&&a.L.va(!0);d(e,f);h.h=0})};
return b}
function ro(a,b,c){b=b.timestamp;return a.now()-b>=c?!1:!0}
function oo(a){if(!mo(a))throw zm("retryQueuedRequests");a.N.ac("QUEUED",a.H).then(function(b){b&&!ro(a,b,a.kc)?a.Y.R(function(){return x(function(c){if(1==c.h)return void 0===b.id?c.u(2):v(c,a.N.Hb(b.id,a.H),2);oo(a);c.h=0})}):a.L.U()&&a.Ra()})}
function qo(a,b){a.zc&&!a.L.U()?a.zc(b):a.handleError(b)}
function mo(a){return!!a.H||a.bb}
function to(a){var b;return(a=null==a?void 0:null==(b=a.error)?void 0:b.code)&&400<=a&&599>=a?!1:!0}
;function uo(a,b){this.version=a;this.args=b}
;function vo(a,b){this.topic=a;this.h=b}
vo.prototype.toString=function(){return this.topic};var wo=B("ytPubsub2Pubsub2Instance")||new K;K.prototype.subscribe=K.prototype.subscribe;K.prototype.unsubscribeByKey=K.prototype.Ga;K.prototype.publish=K.prototype.ra;K.prototype.clear=K.prototype.clear;A("ytPubsub2Pubsub2Instance",wo);var xo=B("ytPubsub2Pubsub2SubscribedKeys")||{};A("ytPubsub2Pubsub2SubscribedKeys",xo);var yo=B("ytPubsub2Pubsub2TopicToKeys")||{};A("ytPubsub2Pubsub2TopicToKeys",yo);var zo=B("ytPubsub2Pubsub2IsAsync")||{};A("ytPubsub2Pubsub2IsAsync",zo);
A("ytPubsub2Pubsub2SkipSubKey",null);function Ao(a,b){var c=Bo();c&&c.publish.call(c,a.toString(),a,b)}
function Co(a){var b=Do,c=Bo();if(!c)return 0;var d=c.subscribe(b.toString(),function(e,f){var g=B("ytPubsub2Pubsub2SkipSubKey");g&&g==d||(g=function(){if(xo[d])try{if(f&&b instanceof vo&&b!=e)try{var h=b.h,k=f;if(!k.args||!k.version)throw Error("yt.pubsub2.Data.deserialize(): serializedData is incomplete.");try{if(!h.oa){var m=new h;h.oa=m.version}var q=h.oa}catch(F){}if(!q||k.version!=q)throw Error("yt.pubsub2.Data.deserialize(): serializedData version is incompatible.");try{q=Reflect;var r=q.construct;
var w=k.args,t=w.length;if(0<t){var y=Array(t);for(k=0;k<t;k++)y[k]=w[k];var E=y}else E=[];f=r.call(q,h,E)}catch(F){throw F.message="yt.pubsub2.Data.deserialize(): "+F.message,F;}}catch(F){throw F.message="yt.pubsub2.pubsub2 cross-binary conversion error for "+b.toString()+": "+F.message,F;}a.call(window,f)}catch(F){Rk(F)}},zo[b.toString()]?B("yt.scheduler.instance")?$h.R(g):kl(g,0):g())});
xo[d]=!0;yo[b.toString()]||(yo[b.toString()]=[]);yo[b.toString()].push(d);return d}
function Eo(){var a=Fo,b=Co(function(c){a.apply(void 0,arguments);Go(b)});
return b}
function Go(a){var b=Bo();b&&("number"===typeof a&&(a=[a]),fb(a,function(c){b.unsubscribeByKey(c);delete xo[c]}))}
function Bo(){return B("ytPubsub2Pubsub2Instance")}
;var Ho;
function Io(){if(Ho)return Ho();var a={};Ho=Kn("LogsDatabaseV2",{Da:(a.LogsRequestsStore={Ia:2},a),pb:!1,upgrade:function(b,c,d){c(2)&&Om(b,"LogsRequestsStore",{keyPath:"id",autoIncrement:!0});c(3);c(5)&&(d=d.objectStore("LogsRequestsStore"),d.h.indexNames.contains("newRequest")&&d.h.deleteIndex("newRequest"),Vm(d,"newRequestV2",["status","interface","timestamp"]));c(7)&&b.h.objectStoreNames.contains("sapisid")&&b.h.deleteObjectStore("sapisid");c(9)&&b.h.objectStoreNames.contains("SWHealthLog")&&b.h.deleteObjectStore("SWHealthLog")},
version:9});return Ho()}
;function Jo(a){return hn(Io(),a)}
function Ko(a,b){var c,d,e,f;return x(function(g){if(1==g.h)return c={startTime:Q(),transactionType:"YT_IDB_TRANSACTION_TYPE_WRITE"},v(g,Jo(b),2);if(3!=g.h)return d=g.i,e=Object.assign({},a,{options:JSON.parse(JSON.stringify(a.options)),interface:L("INNERTUBE_CONTEXT_CLIENT_NAME",0)}),v(g,Qm(d,"LogsRequestsStore",e),3);f=g.i;c.zd=Q();Lo(c);return g.return(f)})}
function Mo(a,b){var c,d,e,f,g,h,k;return x(function(m){if(1==m.h)return c={startTime:Q(),transactionType:"YT_IDB_TRANSACTION_TYPE_READ"},v(m,Jo(b),2);if(3!=m.h)return d=m.i,e=L("INNERTUBE_CONTEXT_CLIENT_NAME",0),f=[a,e,0],g=[a,e,Q()],h=IDBKeyRange.bound(f,g),k=void 0,v(m,Nm(d,["LogsRequestsStore"],{mode:"readwrite",T:!0},function(q){return $m(q.objectStore("LogsRequestsStore").index("newRequestV2"),{query:h,direction:"prev"},function(r){r.getValue()&&(k=r.getValue(),"NEW"===a&&(k.status="QUEUED",
r.update(k)))})}),3);
c.zd=Q();Lo(c);return m.return(k)})}
function No(a,b){var c;return x(function(d){if(1==d.h)return v(d,Jo(b),2);c=d.i;return d.return(Nm(c,["LogsRequestsStore"],{mode:"readwrite",T:!0},function(e){var f=e.objectStore("LogsRequestsStore");return f.get(a).then(function(g){if(g)return g.status="QUEUED",Jm(f.h.put(g,void 0)).then(function(){return g})})}))})}
function Oo(a,b,c){c=void 0===c?!0:c;var d;return x(function(e){if(1==e.h)return v(e,Jo(b),2);d=e.i;return e.return(Nm(d,["LogsRequestsStore"],{mode:"readwrite",T:!0},function(f){var g=f.objectStore("LogsRequestsStore");return g.get(a).then(function(h){return h?(h.status="NEW",c&&(h.sendCount+=1),Jm(g.h.put(h,void 0)).then(function(){return h})):Dm.resolve(void 0)})}))})}
function Po(a,b){var c;return x(function(d){if(1==d.h)return v(d,Jo(b),2);c=d.i;return d.return(c.delete("LogsRequestsStore",a))})}
function Qo(a){var b,c;return x(function(d){if(1==d.h)return v(d,Jo(a),2);b=d.i;c=Q()-2592E6;return v(d,Nm(b,["LogsRequestsStore"],{mode:"readwrite",T:!0},function(e){return Xm(e.objectStore("LogsRequestsStore"),{},function(f){if(f.getValue().timestamp<=c)return f.delete().then(function(){return f.continue()})})}),0)})}
function Ro(){x(function(a){return v(a,Gn(),0)})}
function Lo(a){M("nwl_csi_killswitch")||.01>=Math.random()&&Ao("nwl_transaction_latency_payload",a)}
;var So={},To=Kn("ServiceWorkerLogsDatabase",{Da:(So.SWHealthLog={Ia:1},So),pb:!0,upgrade:function(a,b){b(1)&&Vm(Om(a,"SWHealthLog",{keyPath:"id",autoIncrement:!0}),"swHealthNewRequest",["interface","timestamp"])},
version:1});function Uo(a){return hn(To(),a)}
function Vo(a){var b,c;x(function(d){if(1==d.h)return v(d,Uo(a),2);b=d.i;c=Q()-2592E6;return v(d,Nm(b,["SWHealthLog"],{mode:"readwrite",T:!0},function(e){return Xm(e.objectStore("SWHealthLog"),{},function(f){if(f.getValue().timestamp<=c)return f.delete().then(function(){return f.continue()})})}),0)})}
function Wo(a){var b;return x(function(c){if(1==c.h)return v(c,Uo(a),2);b=c.i;return v(c,b.clear("SWHealthLog"),0)})}
;var Xo={},Yo=0;function Zo(a){var b=new Image,c=""+Yo++;Xo[c]=b;b.onload=b.onerror=function(){delete Xo[c]};
b.src=a}
;function $o(){this.h=new Map;this.i=!1}
function ap(){if(!$o.h){var a=B("yt.networkRequestMonitor.instance")||new $o;A("yt.networkRequestMonitor.instance",a);$o.h=a}return $o.h}
$o.prototype.requestComplete=function(a,b){b&&(this.i=!0);a=this.removeParams(a);this.h.get(a)||this.h.set(a,b)};
$o.prototype.isEndpointCFR=function(a){a=this.removeParams(a);return(a=this.h.get(a))?!1:!1===a&&this.i?!0:null};
$o.prototype.removeParams=function(a){return a.split("?")[0]};
$o.prototype.removeParams=$o.prototype.removeParams;$o.prototype.isEndpointCFR=$o.prototype.isEndpointCFR;$o.prototype.requestComplete=$o.prototype.requestComplete;$o.getInstance=ap;var bp;function cp(){bp||(bp=new am("yt.offline"));return bp}
function dp(a){if(M("offline_error_handling")){var b=cp().get("errors",!0)||{};b[a.message]={name:a.name,stack:a.stack};a.level&&(b[a.message].level=a.level);cp().set("errors",b,2592E3,!0)}}
;function ep(){df.call(this);var a=this;this.j=!1;this.i=Zh();this.i.ha("networkstatus-online",function(){if(a.j&&M("offline_error_handling")){var b=cp().get("errors",!0);if(b){for(var c in b)if(b[c]){var d=new P(c,"sent via offline_errors");d.name=b[c].name;d.stack=b[c].stack;d.level=b[c].level;Rk(d)}cp().set("errors",{},2592E3,!0)}}})}
u(ep,df);function fp(){if(!ep.h){var a=B("yt.networkStatusManager.instance")||new ep;A("yt.networkStatusManager.instance",a);ep.h=a}return ep.h}
l=ep.prototype;l.U=function(){return this.i.U()};
l.va=function(a){this.i.i=a};
l.Sc=function(){var a=window.navigator.onLine;return void 0===a?!0:a};
l.Nc=function(){this.j=!0};
l.ha=function(a,b){return this.i.ha(a,b)};
l.ob=function(a){a=Xh(this.i,a);a.then(function(b){M("use_cfr_monitor")&&ap().requestComplete("generate_204",b)});
return a};
ep.prototype.sendNetworkCheckRequest=ep.prototype.ob;ep.prototype.listen=ep.prototype.ha;ep.prototype.enableErrorFlushing=ep.prototype.Nc;ep.prototype.getWindowStatus=ep.prototype.Sc;ep.prototype.networkStatusHint=ep.prototype.va;ep.prototype.isNetworkAvailable=ep.prototype.U;ep.getInstance=fp;function gp(a){a=void 0===a?{}:a;df.call(this);var b=this;this.i=this.o=0;this.j=fp();var c=B("yt.networkStatusManager.instance.listen").bind(this.j);c&&(a.nb?(this.nb=a.nb,c("networkstatus-online",function(){hp(b,"publicytnetworkstatus-online")}),c("networkstatus-offline",function(){hp(b,"publicytnetworkstatus-offline")})):(c("networkstatus-online",function(){ef(b,"publicytnetworkstatus-online")}),c("networkstatus-offline",function(){ef(b,"publicytnetworkstatus-offline")})))}
u(gp,df);gp.prototype.U=function(){var a=B("yt.networkStatusManager.instance.isNetworkAvailable");return a?a.bind(this.j)():!0};
gp.prototype.va=function(a){var b=B("yt.networkStatusManager.instance.networkStatusHint").bind(this.j);b&&b(a)};
gp.prototype.ob=function(a){var b=this,c;return x(function(d){c=B("yt.networkStatusManager.instance.sendNetworkCheckRequest").bind(b.j);return M("skip_network_check_if_cfr")&&ap().isEndpointCFR("generate_204")?d.return(new Promise(function(e){var f;b.va((null==(f=window.navigator)?void 0:f.onLine)||!0);e(b.U())})):c?d.return(c(a)):d.return(!0)})};
function hp(a,b){a.nb?a.i?($h.ea(a.o),a.o=$h.R(function(){a.m!==b&&(ef(a,b),a.m=b,a.i=Q())},a.nb-(Q()-a.i))):(ef(a,b),a.m=b,a.i=Q()):ef(a,b)}
;var ip;function jp(){var a=lo.call;ip||(ip=new gp({yr:!0,sr:!0}));a.call(lo,this,{N:{Jc:Qo,Ba:Po,ac:Mo,Zc:No,Hb:Oo,set:Ko},L:ip,handleError:Rk,Ca:Sk,ia:kp,now:Q,zc:dp,Y:Wl(),Db:"publicytnetworkstatus-online",Cb:"publicytnetworkstatus-offline",gb:!0,fb:.1,lb:Lk("potential_esf_error_limit",10),J:M,Ka:!(Xl()&&lp())});this.j=new Eh;M("networkless_immediately_drop_all_requests")&&Ro();Hn("LogsDatabaseV2")}
u(jp,lo);function mp(){var a=B("yt.networklessRequestController.instance");a||(a=new jp,A("yt.networklessRequestController.instance",a),M("networkless_logging")&&wn().then(function(b){a.H=b;no(a);a.j.resolve();a.gb&&Math.random()<=a.fb&&a.H&&Vo(a.H);M("networkless_immediately_drop_sw_health_store")&&np(a)}));
return a}
jp.prototype.writeThenSend=function(a,b){b||(b={});Xl()||(this.h=!1);lo.prototype.writeThenSend.call(this,a,b)};
jp.prototype.sendThenWrite=function(a,b,c){b||(b={});Xl()||(this.h=!1);lo.prototype.sendThenWrite.call(this,a,b,c)};
jp.prototype.sendAndWrite=function(a,b){b||(b={});Xl()||(this.h=!1);lo.prototype.sendAndWrite.call(this,a,b)};
jp.prototype.awaitInitialization=function(){return this.j.promise};
function np(a){var b;x(function(c){if(!a.H)throw b=zm("clearSWHealthLogsDb"),b;return c.return(Wo(a.H).catch(function(d){a.handleError(d)}))})}
function kp(a,b,c){M("use_cfr_monitor")&&op(a,b);if(M("use_request_time_ms_header"))b.headers&&(b.headers["X-Goog-Request-Time"]=JSON.stringify(Math.round(Q())));else{var d;if(null==(d=b.postParams)?0:d.requestTimeMs)b.postParams.requestTimeMs=Math.round(Q())}if(c&&0===Object.keys(b).length){var e=void 0===e?"":e;var f=void 0===f?!1:f;if(a)if(e)tl(a,void 0,"POST",e);else if(L("USE_NET_AJAX_FOR_PING_TRANSPORT",!1))tl(a,void 0,"GET","",void 0,void 0,f);else{b:{try{var g=new bb({url:a});if(g.j&&g.i||
g.l){var h=lc(mc(5,a)),k;if(!(k=!h||!h.endsWith("/aclk"))){var m=a.search(xc),q=wc(a,0,"ri",m);if(0>q)var r=null;else{var w=a.indexOf("&",q);if(0>w||w>m)w=m;r=decodeURIComponent(a.slice(q+3,-1!==w?w:0).replace(/\+/g," "))}k="1"!==r}var t=!k;break b}}catch(E){}t=!1}if(t){b:{try{if(window.navigator&&window.navigator.sendBeacon&&window.navigator.sendBeacon(a,"")){var y=!0;break b}}catch(E){}y=!1}c=y?!0:!1}else c=!1;c||Zo(a)}}else ql(a,b)}
function op(a,b){var c=b.onError?b.onError:function(){};
b.onError=function(e,f){ap().requestComplete(a,!1);c(e,f)};
var d=b.onSuccess?b.onSuccess:function(){};
b.onSuccess=function(e,f){ap().requestComplete(a,!0);d(e,f)}}
function lp(){return"www.youtube-nocookie.com"!==nc(document.location.toString())}
;var pp=!1,qp=z.ytNetworklessLoggingInitializationOptions||{isNwlInitialized:pp};A("ytNetworklessLoggingInitializationOptions",qp);function rp(){var a;x(function(b){if(1==b.h)return v(b,wn(),2);a=b.i;if(!a||!Xl()&&!M("nwl_init_require_datasync_id_killswitch")||!lp())return b.u(0);pp=!0;qp.isNwlInitialized=pp;return v(b,mp().awaitInitialization(),0)})}
;function sp(a){var b=this;this.config_=null;a?this.config_=a:Tn()&&(this.config_=Un());Rl(function(){jo(b)},5E3)}
sp.prototype.isReady=function(){!this.config_&&Tn()&&(this.config_=Un());return!!this.config_};
function ko(a,b,c,d){function e(y){y=void 0===y?!1:y;var E;if(d.retry&&"www.youtube-nocookie.com"!=h&&(y||M("skip_ls_gel_retry")||"application/json"!==g.headers["Content-Type"]||(E=ho(b,c,m,k)),E)){var F=g.onSuccess,O=g.onFetchSuccess;g.onSuccess=function(N,R){io(E);F(N,R)};
c.onFetchSuccess=function(N,R){io(E);O(N,R)}}try{y&&d.retry&&!d.hc.bypassNetworkless?(g.method="POST",d.hc.writeThenSend?mp().writeThenSend(t,g):mp().sendAndWrite(t,g)):M("web_all_payloads_via_jspb")?ql(t,g):(g.method="POST",g.postParams||(g.postParams={}),ql(t,g))}catch(N){if("InvalidAccessError"==N.name)E&&(io(E),E=0),Sk(Error("An extension is blocking network request."));
else throw N;}E&&Rl(function(){jo(a)},5E3)}
!L("VISITOR_DATA")&&"visitor_id"!==b&&.01>Math.random()&&Sk(new P("Missing VISITOR_DATA when sending innertube request.",b,c,d));if(!a.isReady()){var f=new P("innertube xhrclient not ready",b,c,d);Rk(f);throw f;}var g={headers:d.headers||{},method:"POST",postParams:c,postBody:d.postBody,postBodyFormat:d.postBodyFormat||"JSON",onTimeout:function(){d.onTimeout()},
onFetchTimeout:d.onTimeout,onSuccess:function(y,E){if(d.onSuccess)d.onSuccess(E)},
onFetchSuccess:function(y){if(d.onSuccess)d.onSuccess(y)},
onError:function(y,E){if(d.onError)d.onError(E)},
onFetchError:function(y){if(d.onError)d.onError(y)},
timeout:d.timeout,withCredentials:!0};g.headers["Content-Type"]||(g.headers["Content-Type"]="application/json");var h="";(f=a.config_.Vc)&&(h=f);var k=a.config_.Xc||!1,m=co(k,h,d);Object.assign(g.headers,m);(f=g.headers.Authorization)&&!h&&(g.headers["x-origin"]=window.location.origin);var q="/youtubei/"+a.config_.innertubeApiVersion+"/"+b,r={alt:"json"},w=a.config_.Wc&&f;w=w&&f.startsWith("Bearer");w||(r.key=a.config_.innertubeApiKey);var t=dl(""+h+q,r||{},!0);(B("ytNetworklessLoggingInitializationOptions")?
qp.isNwlInitialized:pp)?un().then(function(y){e(y)}):e(!1)}
;var tp=0,up=Lc?"webkit":Kc?"moz":Ic?"ms":Hc?"o":"";A("ytDomDomGetNextId",B("ytDomDomGetNextId")||function(){return++tp});var Ip={stopImmediatePropagation:1,stopPropagation:1,preventMouseEvent:1,preventManipulation:1,preventDefault:1,layerX:1,layerY:1,screenX:1,screenY:1,scale:1,rotation:1,webkitMovementX:1,webkitMovementY:1};
function gq(a){this.type="";this.state=this.source=this.data=this.currentTarget=this.relatedTarget=this.target=null;this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.ctrlKey=this.altKey=!1;this.rotation=this.clientY=this.clientX=0;this.scale=1;this.changedTouches=this.touches=null;try{if(a=a||window.event){this.event=a;for(var b in a)b in Ip||(this[b]=a[b]);this.scale=a.scale;this.rotation=a.rotation;var c=a.target||a.srcElement;c&&3==c.nodeType&&(c=c.parentNode);this.target=c;var d=a.relatedTarget;
if(d)try{d=d.nodeName?d:null}catch(e){d=null}else"mouseover"==this.type?d=a.fromElement:"mouseout"==this.type&&(d=a.toElement);this.relatedTarget=d;this.clientX=void 0!=a.clientX?a.clientX:a.pageX;this.clientY=void 0!=a.clientY?a.clientY:a.pageY;this.keyCode=a.keyCode?a.keyCode:a.which;this.charCode=a.charCode||("keypress"==this.type?this.keyCode:0);this.altKey=a.altKey;this.ctrlKey=a.ctrlKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.h=a.pageX;this.i=a.pageY}}catch(e){}}
function hq(a){if(document.body&&document.documentElement){var b=document.body.scrollTop+document.documentElement.scrollTop;a.h=a.clientX+(document.body.scrollLeft+document.documentElement.scrollLeft);a.i=a.clientY+b}}
gq.prototype.preventDefault=function(){this.event&&(this.event.returnValue=!1,this.event.preventDefault&&this.event.preventDefault())};
gq.prototype.stopPropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopPropagation&&this.event.stopPropagation())};
gq.prototype.stopImmediatePropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopImmediatePropagation&&this.event.stopImmediatePropagation())};var ob=z.ytEventsEventsListeners||{};A("ytEventsEventsListeners",ob);var iq=z.ytEventsEventsCounter||{count:0};A("ytEventsEventsCounter",iq);
function jq(a,b,c,d){d=void 0===d?{}:d;a.addEventListener&&("mouseenter"!=b||"onmouseenter"in document?"mouseleave"!=b||"onmouseenter"in document?"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"):b="mouseout":b="mouseover");return nb(function(e){var f="boolean"===typeof e[4]&&e[4]==!!d,g=Qa(e[4])&&Qa(d)&&tb(e[4],d);return!!e.length&&e[0]==a&&e[1]==b&&e[2]==c&&(f||g)})}
var kq=db(function(){var a=!1;try{var b=Object.defineProperty({},"capture",{get:function(){a=!0}});
window.addEventListener("test",null,b)}catch(c){}return a});
function lq(a,b,c,d){d=void 0===d?{}:d;if(!a||!a.addEventListener&&!a.attachEvent)return"";var e=jq(a,b,c,d);if(e)return e;e=++iq.count+"";var f=!("mouseenter"!=b&&"mouseleave"!=b||!a.addEventListener||"onmouseenter"in document);var g=f?function(h){h=new gq(h);if(!of(h.relatedTarget,function(k){return k==a}))return h.currentTarget=a,h.type=b,c.call(a,h)}:function(h){h=new gq(h);
h.currentTarget=a;return c.call(a,h)};
g=Qk(g);a.addEventListener?("mouseenter"==b&&f?b="mouseover":"mouseleave"==b&&f?b="mouseout":"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"),kq()||"boolean"===typeof d?a.addEventListener(b,g,d):a.addEventListener(b,g,!!d.capture)):a.attachEvent("on"+b,g);ob[e]=[a,b,c,g,d];return e}
function mq(a){a&&("string"==typeof a&&(a=[a]),fb(a,function(b){if(b in ob){var c=ob[b],d=c[0],e=c[1],f=c[3];c=c[4];d.removeEventListener?kq()||"boolean"===typeof c?d.removeEventListener(e,f,c):d.removeEventListener(e,f,!!c.capture):d.detachEvent&&d.detachEvent("on"+e,f);delete ob[b]}}))}
;var nq=window.ytcsi&&window.ytcsi.now?window.ytcsi.now:window.performance&&window.performance.timing&&window.performance.now&&window.performance.timing.navigationStart?function(){return window.performance.timing.navigationStart+window.performance.now()}:function(){return(new Date).getTime()};function oq(a){this.F=a;this.i=null;this.m=0;this.s=null;this.o=0;this.j=[];for(a=0;4>a;a++)this.j.push(0);this.l=0;this.K=lq(window,"mousemove",Xa(this.O,this));a=Xa(this.I,this);"function"===typeof a&&(a=Qk(a));this.P=window.setInterval(a,25)}
Za(oq,J);oq.prototype.O=function(a){void 0===a.h&&hq(a);var b=a.h;void 0===a.i&&hq(a);this.i=new kf(b,a.i)};
oq.prototype.I=function(){if(this.i){var a=nq();if(0!=this.m){var b=this.s,c=this.i,d=b.x-c.x;b=b.y-c.y;d=Math.sqrt(d*d+b*b)/(a-this.m);this.j[this.l]=.5<Math.abs((d-this.o)/this.o)?1:0;for(c=b=0;4>c;c++)b+=this.j[c]||0;3<=b&&this.F();this.o=d}this.m=a;this.s=this.i;this.l=(this.l+1)%4}};
oq.prototype.B=function(){window.clearInterval(this.P);mq(this.K)};var pq={};
function qq(a){var b=void 0===a?{}:a;a=void 0===b.ed?!1:b.ed;b=void 0===b.Oc?!0:b.Oc;if(null==B("_lact",window)){var c=parseInt(L("LACT"),10);c=isFinite(c)?Date.now()-Math.max(c,0):-1;A("_lact",c,window);A("_fact",c,window);-1==c&&rq();lq(document,"keydown",rq);lq(document,"keyup",rq);lq(document,"mousedown",rq);lq(document,"mouseup",rq);a?lq(window,"touchmove",function(){sq("touchmove",200)},{passive:!0}):(lq(window,"resize",function(){sq("resize",200)}),b&&lq(window,"scroll",function(){sq("scroll",200)}));
new oq(function(){sq("mouse",100)});
lq(document,"touchstart",rq,{passive:!0});lq(document,"touchend",rq,{passive:!0})}}
function sq(a,b){pq[a]||(pq[a]=!0,$h.R(function(){rq();pq[a]=!1},b))}
function rq(){null==B("_lact",window)&&qq();var a=Date.now();A("_lact",a,window);-1==B("_fact",window)&&A("_fact",a,window);(a=B("ytglobal.ytUtilActivityCallback_"))&&a()}
function tq(){var a=B("_lact",window);return null==a?-1:Math.max(Date.now()-a,0)}
;var uq=z.ytPubsubPubsubInstance||new K,vq=z.ytPubsubPubsubSubscribedKeys||{},wq=z.ytPubsubPubsubTopicToKeys||{},xq=z.ytPubsubPubsubIsSynchronous||{};function yq(a,b){var c=zq();if(c&&b){var d=c.subscribe(a,function(){var e=arguments;var f=function(){vq[d]&&b.apply&&"function"==typeof b.apply&&b.apply(window,e)};
try{xq[a]?f():kl(f,0)}catch(g){Rk(g)}},void 0);
vq[d]=!0;wq[a]||(wq[a]=[]);wq[a].push(d);return d}return 0}
function Aq(a){var b=zq();b&&("number"===typeof a?a=[a]:"string"===typeof a&&(a=[parseInt(a,10)]),fb(a,function(c){b.unsubscribeByKey(c);delete vq[c]}))}
function Bq(a,b){var c=zq();c&&c.publish.apply(c,arguments)}
function Cq(a){var b=zq();if(b)if(b.clear(a),a)Dq(a);else for(var c in wq)Dq(c)}
function zq(){return z.ytPubsubPubsubInstance}
function Dq(a){wq[a]&&(a=wq[a],fb(a,function(b){vq[b]&&delete vq[b]}),a.length=0)}
K.prototype.subscribe=K.prototype.subscribe;K.prototype.unsubscribeByKey=K.prototype.Ga;K.prototype.publish=K.prototype.ra;K.prototype.clear=K.prototype.clear;A("ytPubsubPubsubInstance",uq);A("ytPubsubPubsubTopicToKeys",wq);A("ytPubsubPubsubIsSynchronous",xq);A("ytPubsubPubsubSubscribedKeys",vq);var Eq=Symbol("injectionDeps");function Fq(a){this.name=a}
Fq.prototype.toString=function(){return"InjectionToken("+this.name+")"};
function Gq(){this.key=Hq}
function Iq(){this.h=new Map;this.i=new Map}
Iq.prototype.resolve=function(a){return a instanceof Gq?Jq(this,a.key,[],!0):Jq(this,a,[])};
function Jq(a,b,c,d){d=void 0===d?!1:d;if(-1<c.indexOf(b))throw Error("Deps cycle for: "+b);if(a.i.has(b))return a.i.get(b);if(!a.h.has(b)){if(d)return;throw Error("No provider for: "+b);}d=a.h.get(b);c.push(b);if(d.xc)var e=d.xc;else if(d.Cd)e=d[Eq]?Kq(a,d[Eq],c):[],e=d.Cd.apply(d,ia(e));else if(d.wc){e=d.wc;var f=e[Eq]?Kq(a,e[Eq],c):[];e=new (Function.prototype.bind.apply(e,[null].concat(ia(f))))}else throw Error("Could not resolve providers for: "+b);c.pop();d.Ir||a.i.set(b,e);return e}
function Kq(a,b,c){return b?b.map(function(d){return d instanceof Gq?Jq(a,d.key,c,!0):Jq(a,d,c)}):[]}
;var Lq;function Mq(){Lq||(Lq=new Iq);return Lq}
;function Nq(){this.store={};this.h={}}
Nq.prototype.storePayload=function(a,b){a=Oq(a);this.store[a]?this.store[a].push(b):(this.h={},this.store[a]=[b]);return a};
Nq.prototype.extractMatchingEntries=function(a){a=Pq(this,a);for(var b=[],c=0;c<a.length;c++)this.store[a[c]]&&(b.push.apply(b,ia(this.store[a[c]])),delete this.store[a[c]]);return b};
Nq.prototype.getSequenceCount=function(a){a=Pq(this,a);for(var b=0,c=0;c<a.length;c++)b+=this.store[a[c]].length||0;return b};
function Pq(a,b){var c=Oq(b);if(a.h[c])return a.h[c];var d=Object.keys(a.store)||[];if(1>=d.length&&Oq(b)===d[0])return d;for(var e=[],f=0;f<d.length;f++){var g=d[f].split("/");if(Qq(b.auth,g[0])){var h=b.isJspb;Qq(void 0===h?"undefined":h?"true":"false",g[1])&&Qq(b.cttAuthInfo,g[2])&&e.push(d[f])}}return a.h[c]=e}
function Qq(a,b){return void 0===a||"undefined"===a?!0:a===b}
Nq.prototype.getSequenceCount=Nq.prototype.getSequenceCount;Nq.prototype.extractMatchingEntries=Nq.prototype.extractMatchingEntries;Nq.prototype.storePayload=Nq.prototype.storePayload;function Oq(a){return[void 0===a.auth?"undefined":a.auth,void 0===a.isJspb?"undefined":a.isJspb,void 0===a.cttAuthInfo?"undefined":a.cttAuthInfo].join("/")}
;function Rq(a,b){if(a)return a[b.name]}
;var Sq=Lk("initial_gel_batch_timeout",2E3),Tq=Math.pow(2,16)-1,Uq=void 0;function Vq(){this.j=this.h=this.i=0}
var Wq=new Vq,Xq=new Vq,Yq,Zq=!0,$q=z.ytLoggingTransportGELQueue_||new Map;A("ytLoggingTransportGELQueue_",$q);var ar=z.ytLoggingTransportGELProtoQueue_||new Map;A("ytLoggingTransportGELProtoQueue_",ar);var br=z.ytLoggingTransportTokensToCttTargetIds_||{};A("ytLoggingTransportTokensToCttTargetIds_",br);var cr=z.ytLoggingTransportTokensToJspbCttTargetIds_||{};A("ytLoggingTransportTokensToJspbCttTargetIds_",cr);var dr={};function er(){var a=B("yt.logging.ims");a||(a=new Nq,A("yt.logging.ims",a));return a}
function fr(a,b){M("web_all_payloads_via_jspb")&&Sk(new P("transport.log called for JSON in JSPB only experiment"));if("log_event"===a.endpoint){gr(a);var c=hr(a);if(M("use_new_in_memory_storage")){dr[c]=!0;var d={cttAuthInfo:c,isJspb:!1};er().storePayload(d,a.payload);ir(b,[],c,!1,d)}else d=$q.get(c)||[],$q.set(c,d),d.push(a.payload),ir(b,d,c)}}
function jr(a,b){if("log_event"===a.endpoint){gr(void 0,a);var c=hr(a,!0);if(M("use_new_in_memory_storage")){dr[c]=!0;var d={cttAuthInfo:c,isJspb:!0};er().storePayload(d,a.payload.toJSON());ir(b,[],c,!0,d)}else d=ar.get(c)||[],ar.set(c,d),a=a.payload.toJSON(),d.push(a),ir(b,d,c,!0)}}
function ir(a,b,c,d,e){d=void 0===d?!1:d;a&&(Uq=new a);a=Lk("tvhtml5_logging_max_batch")||Lk("web_logging_max_batch")||100;var f=Q(),g=d?Xq.j:Wq.j;b=b.length;e&&(b=er().getSequenceCount(e));b>=a?Yq||(Yq=kr(function(){lr({writeThenSend:!0},M("flush_only_full_queue")?c:void 0,d);Yq=void 0},0)):10<=f-g&&(mr(d),d?Xq.j=f:Wq.j=f)}
function nr(a,b){M("web_all_payloads_via_jspb")&&Sk(new P("transport.logIsolatedGelPayload called in JSPB only experiment"));if("log_event"===a.endpoint){gr(a);var c=hr(a),d=new Map;d.set(c,[a.payload]);b&&(Uq=new b);return new Af(function(e,f){Uq&&Uq.isReady()?or(d,Uq,e,f,{bypassNetworkless:!0},!0):e()})}}
function pr(a,b){if("log_event"===a.endpoint){gr(void 0,a);var c=hr(a,!0),d=new Map;d.set(c,[a.payload.toJSON()]);b&&(Uq=new b);return new Af(function(e){Uq&&Uq.isReady()?qr(d,Uq,e,{bypassNetworkless:!0},!0):e()})}}
function hr(a,b){var c="";if(a.Ja)c="visitorOnlyApprovedKey";else if(a.cttAuthInfo){if(void 0===b?0:b){b=a.cttAuthInfo.token;c=a.cttAuthInfo;var d=new ak;c.videoId?d.setVideoId(c.videoId):c.playlistId&&Vd(d,2,bk,c.playlistId);cr[b]=d}else b=a.cttAuthInfo,c={},b.videoId?c.videoId=b.videoId:b.playlistId&&(c.playlistId=b.playlistId),br[a.cttAuthInfo.token]=c;c=a.cttAuthInfo.token}return c}
function lr(a,b,c){a=void 0===a?{}:a;c=void 0===c?!1:c;!c&&M("web_all_payloads_via_jspb")&&Sk(new P("transport.flushLogs called for JSON in JSPB only experiment"));new Af(function(d,e){c?(rr(Xq.i),rr(Xq.h),Xq.h=0):(rr(Wq.i),rr(Wq.h),Wq.h=0);if(Uq&&Uq.isReady())if(M("use_new_in_memory_storage")){var f=a,g=c,h=Uq;f=void 0===f?{}:f;g=void 0===g?!1:g;var k=new Map,m=new Map;if(void 0!==b)g?(e=er().extractMatchingEntries({isJspb:g,cttAuthInfo:b}),k.set(b,e),qr(k,h,d,f)):(k=er().extractMatchingEntries({isJspb:g,
cttAuthInfo:b}),m.set(b,k),or(m,h,d,e,f));else if(g){e=p(Object.keys(dr));for(g=e.next();!g.done;g=e.next())m=g.value,g=er().extractMatchingEntries({isJspb:!0,cttAuthInfo:m}),0<g.length&&k.set(m,g),delete dr[m];qr(k,h,d,f)}else{k=p(Object.keys(dr));for(g=k.next();!g.done;g=k.next()){g=g.value;var q=er().extractMatchingEntries({isJspb:!1,cttAuthInfo:g});0<q.length&&m.set(g,q);delete dr[g]}or(m,h,d,e,f)}}else f=a,k=c,h=Uq,f=void 0===f?{}:f,k=void 0===k?!1:k,void 0!==b?k?(e=new Map,k=ar.get(b)||[],e.set(b,
k),qr(e,h,d,f),ar.delete(b)):(k=new Map,m=$q.get(b)||[],k.set(b,m),or(k,h,d,e,f),$q.delete(b)):k?(qr(ar,h,d,f),ar.clear()):(or($q,h,d,e,f),$q.clear());else mr(c),d()})}
function mr(a){a=void 0===a?!1:a;if(M("web_gel_timeout_cap")&&(!a&&!Wq.h||a&&!Xq.h)){var b=kr(function(){lr({writeThenSend:!0},void 0,a)},6E4);
a?Xq.h=b:Wq.h=b}rr(a?Xq.i:Wq.i);b=L("LOGGING_BATCH_TIMEOUT",Lk("web_gel_debounce_ms",1E4));M("shorten_initial_gel_batch_timeout")&&Zq&&(b=Sq);b=kr(function(){lr({writeThenSend:!0},void 0,a)},b);
a?Xq.i=b:Wq.i=b}
function or(a,b,c,d,e,f){e=void 0===e?{}:e;var g=Math.round(Q()),h=a.size,k={};a=p(a);for(var m=a.next();!m.done;k={Ua:k.Ua,qa:k.qa,Ea:k.Ea,Wa:k.Wa,Va:k.Va},m=a.next()){var q=p(m.value);m=q.next().value;q=q.next().value;k.qa=vb({context:Vn(b.config_||Un())});if(!Pa(q)&&!M("throw_err_when_logevent_malformed_killswitch")){d();break}k.qa.events=q;(q=br[m])&&sr(k.qa,m,q);delete br[m];k.Ea="visitorOnlyApprovedKey"===m;tr(k.qa,g,k.Ea);ur(e);k.Wa=function(r){M("update_log_event_config")&&$h.R(function(){return x(function(w){return v(w,
vr(r),0)})});
h--;h||c()};
k.Ua=0;k.Va=function(r){return function(){r.Ua++;if(e.bypassNetworkless&&1===r.Ua)try{ko(b,"log_event",r.qa,wr({writeThenSend:!0},r.Ea,r.Wa,r.Va,f)),Zq=!1}catch(w){Rk(w),d()}h--;h||c()}}(k);
try{ko(b,"log_event",k.qa,wr(e,k.Ea,k.Wa,k.Va,f)),Zq=!1}catch(r){Rk(r),d()}}}
function qr(a,b,c,d,e){d=void 0===d?{}:d;var f=Math.round(Q()),g=a.size,h=new Map([].concat(ia(a)));h=p(h);for(var k=h.next();!k.done;k=h.next()){var m=p(k.value).next().value,q=a.get(m);k=new ck;var r=bo(b.config_||Un());G(k,fj,1,r);q=q?xr(q):[];q=p(q);for(r=q.next();!r.done;r=q.next())ce(k,3,Wj,r.value);(q=cr[m])&&yr(k,m,q);delete cr[m];m="visitorOnlyApprovedKey"===m;zr(k,f,m);ur(d);k=ke(k);m=wr(d,m,function(w){M("update_log_event_config")&&$h.R(function(){return x(function(t){return v(t,vr(w),
0)})});
g--;g||c()},function(){g--;
g||c()},e);
m.headers["Content-Type"]="application/json+protobuf";m.postBodyFormat="JSPB";m.postBody=k;ko(b,"log_event","",m);Zq=!1}}
function ur(a){M("always_send_and_write")&&(a.writeThenSend=!1)}
function wr(a,b,c,d,e){a={retry:!0,onSuccess:c,onError:d,hc:a,Ja:b,mr:!!e,headers:{},postBodyFormat:"",postBody:""};Ar()&&(a.headers["X-Goog-Request-Time"]=JSON.stringify(Math.round(Q())));return a}
function tr(a,b,c){Ar()||(a.requestTimeMs=String(b));M("unsplit_gel_payloads_in_logs")&&(a.unsplitGelPayloadsInLogs=!0);!c&&(b=L("EVENT_ID"))&&(c=Br(),a.serializedClientEventId={serializedEventId:b,clientCounter:String(c)})}
function zr(a,b,c){Ar()||D(a,2,b);if(!c&&(b=L("EVENT_ID"))){c=Br();var d=new Zj;D(d,1,b);D(d,2,c);G(a,Zj,5,d)}}
function Br(){var a=L("BATCH_CLIENT_COUNTER")||0;a||(a=Math.floor(Math.random()*Tq/2));a++;a>Tq&&(a=1);Ik("BATCH_CLIENT_COUNTER",a);return a}
function sr(a,b,c){if(c.videoId)var d="VIDEO";else if(c.playlistId)d="PLAYLIST";else return;a.credentialTransferTokenTargetId=c;a.context=a.context||{};a.context.user=a.context.user||{};a.context.user.credentialTransferTokens=[{token:b,scope:d}]}
function yr(a,b,c){if(de(c,1===Wd(c,bk)?1:-1))var d=1;else if(c.getPlaylistId())d=2;else return;G(a,ak,4,c);a=Xd(a,fj,1)||new fj;c=Xd(a,dj,3)||new dj;var e=new cj;D(e,2,b);D(e,1,d);ce(c,12,cj,e);G(a,dj,3,c)}
function xr(a){for(var b=[],c=0;c<a.length;c++)try{b.push(new Wj(a[c]))}catch(d){Rk(new P("Transport failed to deserialize "+String(a[c])))}return b}
function gr(a,b){if(B("yt.logging.transport.enableScrapingForTest")){var c=B("yt.logging.transport.scrapedPayloadsForTesting"),d=B("yt.logging.transport.payloadToScrape","");b&&(b=B("yt.logging.transport.getScrapedPayloadFromClientEventsFunction").bind(b.payload)())&&c.push(b);a&&a.payload[d]&&c.push((null==a?void 0:a.payload)[d]);A("yt.logging.transport.scrapedPayloadsForTesting",c)}}
function Ar(){return M("use_request_time_ms_header")||M("lr_use_request_time_ms_header")}
function kr(a,b){return M("transport_use_scheduler")?Rl(a,b):kl(a,b)}
function rr(a){M("transport_use_scheduler")?$h.ea(a):window.clearTimeout(a)}
function vr(a){var b,c,d,e,f,g,h,k,m,q;return x(function(r){return 1==r.h?(d=null==(b=a)?void 0:null==(c=b.responseContext)?void 0:c.globalConfigGroup,e=Rq(d,Pi),g=null==(f=d)?void 0:f.hotHashData,h=Rq(d,Oi),m=null==(k=d)?void 0:k.coldHashData,q=Mq().resolve(Qn),e&&g?v(r,Rn(q,e,g),2):r.u(2)):h&&m?v(r,Sn(q,h,m),0):r.u(0)})}
;var Cr=z.ytLoggingGelSequenceIdObj_||{};A("ytLoggingGelSequenceIdObj_",Cr);
function Dr(a,b,c,d){d=void 0===d?{}:d;var e={},f=Math.round(d.timestamp||Q());e.eventTimeMs=f<Number.MAX_SAFE_INTEGER?f:0;e[a]=b;M("enable_unknown_lact_fix_on_html5")&&qq();a=tq();e.context={lastActivityMs:String(d.timestamp||!isFinite(a)?-1:a)};M("log_sequence_info_on_gel_web")&&d.Z&&(a=e.context,b=d.Z,b={index:Er(b),groupKey:b},a.sequence=b,d.Xb&&delete Cr[d.Z]);(d.pc?nr:fr)({endpoint:"log_event",payload:e,cttAuthInfo:d.cttAuthInfo,Ja:d.Ja},c)}
function Fr(a){lr(void 0,void 0,void 0===a?!1:a)}
function Er(a){Cr[a]=a in Cr?Cr[a]+1:0;return Cr[a]}
;var Gr=[];var Hr=z.ytLoggingGelSequenceIdObj_||{};A("ytLoggingGelSequenceIdObj_",Hr);function Ir(a,b,c){c=void 0===c?{}:c;var d=Math.round(c.timestamp||Q());D(a,1,d<Number.MAX_SAFE_INTEGER?d:0);var e=tq();d=new Vj;D(d,1,c.timestamp||!isFinite(e)?-1:e);if(M("log_sequence_info_on_gel_web")&&c.Z){e=c.Z;var f=Er(e),g=new Uj;D(g,2,f);D(g,1,e);G(d,Uj,3,g);c.Xb&&delete Hr[c.Z]}G(a,Vj,33,d);(c.pc?pr:jr)({endpoint:"log_event",payload:a,cttAuthInfo:c.cttAuthInfo,Ja:c.Ja},b)}
;function Jr(a,b){b=void 0===b?{}:b;var c=!1;L("ytLoggingEventsDefaultDisabled",!1)&&(c=!0);Ir(a,c?null:sp,b)}
;function Kr(a,b){var c=new Wj;ae(c,Lj,72,Xj,a);Jr(c,b)}
function Lr(a,b,c){var d=new Wj;ae(d,Kj,73,Xj,a);c?Ir(d,c,b):Jr(d,b)}
function Mr(a,b,c){var d=new Wj;ae(d,Jj,78,Xj,a);c?Ir(d,c,b):Jr(d,b)}
function Nr(a,b,c){var d=new Wj;ae(d,Mj,208,Xj,a);c?Ir(d,c,b):Jr(d,b)}
function Or(a,b,c){var d=new Wj;ae(d,Cj,156,Xj,a);c?Ir(d,c,b):Jr(d,b)}
function Pr(a,b,c){var d=new Wj;ae(d,Gj,215,Xj,a);c?Ir(d,c,b):Jr(d,b)}
function Qr(a,b,c){var d=new Wj;ae(d,Bj,111,Xj,a);c?Ir(d,c,b):Jr(d,b)}
;function jm(a,b,c){c=void 0===c?{}:c;if(M("migrate_events_to_ts")){c=void 0===c?{}:c;c.timestamp=Math.round(c.timestamp||Q());var d=sp;L("ytLoggingEventsDefaultDisabled",!1)&&sp===sp&&(d=null);M("web_all_payloads_via_jspb")?Gr.push({Eb:a,payload:b,options:c}):Dr(a,b,d,c)}else d=sp,L("ytLoggingEventsDefaultDisabled",!1)&&sp==sp&&(d=null),Dr(a,b,d,c)}
;var Rr=[{Bb:function(a){return"Cannot read property '"+a.key+"'"},
kb:{Error:[{regexp:/(Permission denied) to access property "([^']+)"/,groups:["reason","key"]}],TypeError:[{regexp:/Cannot read property '([^']+)' of (null|undefined)/,groups:["key","value"]},{regexp:/\u65e0\u6cd5\u83b7\u53d6\u672a\u5b9a\u4e49\u6216 (null|undefined) \u5f15\u7528\u7684\u5c5e\u6027\u201c([^\u201d]+)\u201d/,groups:["value","key"]},{regexp:/\uc815\uc758\ub418\uc9c0 \uc54a\uc74c \ub610\ub294 (null|undefined) \ucc38\uc870\uc778 '([^']+)' \uc18d\uc131\uc744 \uac00\uc838\uc62c \uc218 \uc5c6\uc2b5\ub2c8\ub2e4./,
groups:["value","key"]},{regexp:/No se puede obtener la propiedad '([^']+)' de referencia nula o sin definir/,groups:["key"]},{regexp:/Unable to get property '([^']+)' of (undefined or null) reference/,groups:["key","value"]},{regexp:/(null) is not an object \(evaluating '(?:([^.]+)\.)?([^']+)'\)/,groups:["value","base","key"]}]}},{Bb:function(a){return"Cannot call '"+a.key+"'"},
kb:{TypeError:[{regexp:/(?:([^ ]+)?\.)?([^ ]+) is not a function/,groups:["base","key"]},{regexp:/([^ ]+) called on (null or undefined)/,groups:["key","value"]},{regexp:/Object (.*) has no method '([^ ]+)'/,groups:["base","key"]},{regexp:/Object doesn't support property or method '([^ ]+)'/,groups:["key"]},{regexp:/\u30aa\u30d6\u30b8\u30a7\u30af\u30c8\u306f '([^']+)' \u30d7\u30ed\u30d1\u30c6\u30a3\u307e\u305f\u306f\u30e1\u30bd\u30c3\u30c9\u3092\u30b5\u30dd\u30fc\u30c8\u3057\u3066\u3044\u307e\u305b\u3093/,
groups:["key"]},{regexp:/\uac1c\uccb4\uac00 '([^']+)' \uc18d\uc131\uc774\ub098 \uba54\uc11c\ub4dc\ub97c \uc9c0\uc6d0\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4./,groups:["key"]}]}},{Bb:function(a){return a.key+" is not defined"},
kb:{ReferenceError:[{regexp:/(.*) is not defined/,groups:["key"]},{regexp:/Can't find variable: (.*)/,groups:["key"]}]}}];var Tr={na:[],ka:[{callback:Sr,weight:500}]};function Sr(a){if("JavaException"===a.name)return!0;a=a.stack;return a.includes("chrome://")||a.includes("chrome-extension://")||a.includes("moz-extension://")}
;function Ur(){this.ka=[];this.na=[]}
var Vr;function Wr(){if(!Vr){var a=Vr=new Ur;a.na.length=0;a.ka.length=0;Tr.na&&a.na.push.apply(a.na,Tr.na);Tr.ka&&a.ka.push.apply(a.ka,Tr.ka)}return Vr}
;var Xr=new K;function Yr(a){function b(){return a.charCodeAt(d++)}
var c=a.length,d=0;do{var e=Zr(b);if(Infinity===e)break;var f=e>>3;switch(e&7){case 0:e=Zr(b);if(2===f)return e;break;case 1:if(2===f)return;d+=8;break;case 2:e=Zr(b);if(2===f)return a.substr(d,e);d+=e;break;case 5:if(2===f)return;d+=4;break;default:return}}while(d<c)}
function Zr(a){var b=a(),c=b&127;if(128>b)return c;b=a();c|=(b&127)<<7;if(128>b)return c;b=a();c|=(b&127)<<14;if(128>b)return c;b=a();return 128>b?c|(b&127)<<21:Infinity}
;function $r(a,b,c,d){if(a)if(Array.isArray(a)){var e=d;for(d=0;d<a.length&&!(a[d]&&(e+=as(d,a[d],b,c),500<e));d++);d=e}else if("object"===typeof a)for(e in a){if(a[e]){var f=a[e];var g=b;var h=c;g="string"!==typeof f||"clickTrackingParams"!==e&&"trackingParams"!==e?0:(f=Yr(atob(f.replace(/-/g,"+").replace(/_/g,"/"))))?as(e+".ve",f,g,h):0;d+=g;d+=as(e,a[e],b,c);if(500<d)break}}else c[b]=bs(a),d+=c[b].length;else c[b]=bs(a),d+=c[b].length;return d}
function as(a,b,c,d){c+="."+a;a=bs(b);d[c]=a;return c.length+a.length}
function bs(a){try{return("string"===typeof a?a:String(JSON.stringify(a))).substr(0,500)}catch(b){return"unable to serialize "+typeof a+" ("+b.message+")"}}
;var cs=new Set,ds=0,es=0,fs=0,gs=[],hs=["PhantomJS","Googlebot","TO STOP THIS SECURITY SCAN go/scan"];function im(a){is(a)}
function js(a){is(a,"WARNING")}
function is(a,b,c,d,e,f){f=void 0===f?{}:f;f.name=c||L("INNERTUBE_CONTEXT_CLIENT_NAME",1);f.version=d||L("INNERTUBE_CONTEXT_CLIENT_VERSION");var g=f||{},h=void 0===b?"ERROR":b;h=void 0===h?"ERROR":h;if(a){a.hasOwnProperty("level")&&a.level&&(h=a.level);if(M("console_log_js_exceptions")){var k=[];k.push("Name: "+a.name);k.push("Message: "+a.message);a.hasOwnProperty("params")&&k.push("Error Params: "+JSON.stringify(a.params));a.hasOwnProperty("args")&&k.push("Error args: "+JSON.stringify(a.args));
k.push("File name: "+a.fileName);k.push("Stacktrace: "+a.stack);window.console.log(k.join("\n"),a)}if(!(5<=ds)){var m=gs,q=xe(a),r=q.message||"Unknown Error",w=q.name||"UnknownError",t=q.stack||a.i||"Not available";if(t.startsWith(w+": "+r)){var y=t.split("\n");y.shift();t=y.join("\n")}var E=q.lineNumber||"Not available",F=q.fileName||"Not available",O=t,N=0;if(a.hasOwnProperty("args")&&a.args&&a.args.length)for(var R=0;R<a.args.length&&!(N=$r(a.args[R],"params."+R,g,N),500<=N);R++);else if(a.hasOwnProperty("params")&&
a.params){var ca=a.params;if("object"===typeof a.params)for(var U in ca){if(ca[U]){var rb="params."+U,Va=bs(ca[U]);g[rb]=Va;N+=rb.length+Va.length;if(500<N)break}}else g.params=bs(ca)}if(m.length)for(var na=0;na<m.length&&!(N=$r(m[na],"params.context."+na,g,N),500<=N);na++);navigator.vendor&&!g.hasOwnProperty("vendor")&&(g["device.vendor"]=navigator.vendor);var I={message:r,name:w,lineNumber:E,fileName:F,stack:O,params:g,sampleWeight:1},ma=Number(a.columnNumber);isNaN(ma)||(I.lineNumber=I.lineNumber+
":"+ma);if("IGNORED"===a.level)var ea=0;else a:{for(var Ee=Wr(),Fe=p(Ee.na),sd=Fe.next();!sd.done;sd=Fe.next()){var pa=sd.value;if(I.message&&I.message.match(pa.zr)){ea=pa.weight;break a}}for(var vp=p(Ee.ka),gk=vp.next();!gk.done;gk=vp.next()){var wp=gk.value;if(wp.callback(I)){ea=wp.weight;break a}}ea=1}I.sampleWeight=ea;for(var xp=p(Rr),hk=xp.next();!hk.done;hk=xp.next()){var ik=hk.value;if(ik.kb[I.name])for(var yp=p(ik.kb[I.name]),jk=yp.next();!jk.done;jk=yp.next()){var zp=jk.value,Sg=I.message.match(zp.regexp);
if(Sg){I.params["params.error.original"]=Sg[0];for(var kk=zp.groups,Ap={},td=0;td<kk.length;td++)Ap[kk[td]]=Sg[td+1],I.params["params.error."+kk[td]]=Sg[td+1];I.message=ik.Bb(Ap);break}}}I.params||(I.params={});var Bp=Wr();I.params["params.errorServiceSignature"]="msg="+Bp.na.length+"&cb="+Bp.ka.length;I.params["params.serviceWorker"]="false";z.document&&z.document.querySelectorAll&&(I.params["params.fscripts"]=String(document.querySelectorAll("script:not([nonce])").length));Cb("sample").constructor!==
Ab&&(I.params["params.fconst"]="true");window.yterr&&"function"===typeof window.yterr&&window.yterr(I);if(0!==I.sampleWeight&&!cs.has(I.message)){if("ERROR"===h){Xr.ra("handleError",I);if(M("record_app_crashed_web")&&0===fs&&1===I.sampleWeight)if(fs++,M("errors_via_jspb")){var lk=new yj;D(lk,1,1);if(!M("report_client_error_with_app_crash_ks")){var Cp=new tj;D(Cp,1,I.message);var Dp=new uj;G(Dp,tj,3,Cp);var Ep=new vj;G(Ep,uj,5,Dp);var Fp=new wj;G(Fp,vj,9,Ep);G(lk,wj,4,Fp)}var Gp=new Wj;ae(Gp,yj,20,
Xj,lk);Jr(Gp)}else{var Hp={appCrashType:"APP_CRASH_TYPE_BREAKPAD"};M("report_client_error_with_app_crash_ks")||(Hp.systemHealth={crashData:{clientError:{logMessage:{message:I.message}}}});jm("appCrashed",Hp)}es++}else"WARNING"===h&&Xr.ra("handleWarning",I);if(M("kevlar_gel_error_routing"))a:{var Ge=h;if(M("errors_via_jspb")){if(ks())var Jp=void 0;else{var ud=new qj;D(ud,1,I.stack);I.fileName&&D(ud,4,I.fileName);var Lb=I.lineNumber&&I.lineNumber.split?I.lineNumber.split(":"):[];0!==Lb.length&&(1!==
Lb.length||isNaN(Number(Lb[0]))?2!==Lb.length||isNaN(Number(Lb[0]))||isNaN(Number(Lb[1]))||(D(ud,2,Number(Lb[0])),D(ud,3,Number(Lb[1]))):D(ud,2,Number(Lb[0])));var zc=new tj;D(zc,1,I.message);D(zc,3,I.name);D(zc,6,I.sampleWeight);"ERROR"===Ge?D(zc,2,2):"WARNING"===Ge?D(zc,2,1):D(zc,2,0);var mk=new rj;D(mk,1,!0);ae(mk,qj,3,sj,ud);var bc=new nj;D(bc,3,window.location.href);for(var Kp=L("FEXP_EXPERIMENTS",[]),nk=0;nk<Kp.length;nk++){var cw=Kp[nk];Fd(bc);Ud(bc,5,2,!1,!1).push(cw)}var ok=L("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS");
if(!Jk()&&ok)for(var Lp=p(Object.keys(ok)),Ac=Lp.next();!Ac.done;Ac=Lp.next()){var Mp=Ac.value,pk=new pj;D(pk,1,Mp);D(pk,2,String(ok[Mp]));ce(bc,4,pj,pk)}var qk=I.params;if(qk){var Np=p(Object.keys(qk));for(Ac=Np.next();!Ac.done;Ac=Np.next()){var Op=Ac.value,rk=new pj;D(rk,1,"client."+Op);D(rk,2,String(qk[Op]));ce(bc,4,pj,rk)}}var Pp=L("SERVER_NAME"),Qp=L("SERVER_VERSION");if(Pp&&Qp){var sk=new pj;D(sk,1,"server.name");D(sk,2,Pp);ce(bc,4,pj,sk);var tk=new pj;D(tk,1,"server.version");D(tk,2,Qp);ce(bc,
4,pj,tk)}var Tg=new uj;G(Tg,nj,1,bc);G(Tg,rj,2,mk);G(Tg,tj,3,zc);Jp=Tg}var Rp=Jp;if(!Rp)break a;var Sp=new Wj;ae(Sp,uj,163,Xj,Rp);Jr(Sp)}else{if(ks())var Tp=void 0;else{var He={stackTrace:I.stack};I.fileName&&(He.filename=I.fileName);var Mb=I.lineNumber&&I.lineNumber.split?I.lineNumber.split(":"):[];0!==Mb.length&&(1!==Mb.length||isNaN(Number(Mb[0]))?2!==Mb.length||isNaN(Number(Mb[0]))||isNaN(Number(Mb[1]))||(He.lineNumber=Number(Mb[0]),He.columnNumber=Number(Mb[1])):He.lineNumber=Number(Mb[0]));
var uk={level:"ERROR_LEVEL_UNKNOWN",message:I.message,errorClassName:I.name,sampleWeight:I.sampleWeight};"ERROR"===Ge?uk.level="ERROR_LEVEL_ERROR":"WARNING"===Ge&&(uk.level="ERROR_LEVEL_WARNNING");var dw={isObfuscated:!0,browserStackInfo:He},vd={pageUrl:window.location.href,kvPairs:[]};L("FEXP_EXPERIMENTS")&&(vd.experimentIds=L("FEXP_EXPERIMENTS"));var vk=L("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS");if(!Jk()&&vk)for(var Up=p(Object.keys(vk)),Bc=Up.next();!Bc.done;Bc=Up.next()){var Vp=Bc.value;vd.kvPairs.push({key:Vp,
value:String(vk[Vp])})}var wk=I.params;if(wk){var Wp=p(Object.keys(wk));for(Bc=Wp.next();!Bc.done;Bc=Wp.next()){var Xp=Bc.value;vd.kvPairs.push({key:"client."+Xp,value:String(wk[Xp])})}}var Yp=L("SERVER_NAME"),Zp=L("SERVER_VERSION");Yp&&Zp&&(vd.kvPairs.push({key:"server.name",value:Yp}),vd.kvPairs.push({key:"server.version",value:Zp}));Tp={errorMetadata:vd,stackTrace:dw,logMessage:uk}}var $p=Tp;if(!$p)break a;jm("clientError",$p)}if("ERROR"===Ge||M("errors_flush_gel_always_killswitch"))b:if(M("migrate_events_to_ts"))c:{if(M("web_fp_via_jspb")&&
(Fr(!0),!M("web_fp_via_jspb_and_json")))break c;Fr()}else{if(M("web_fp_via_jspb")&&(Fr(!0),!M("web_fp_via_jspb_and_json")))break b;Fr()}}if(!M("suppress_error_204_logging")){var Ie=I.params||{},cc={urlParams:{a:"logerror",t:"jserror",type:I.name,msg:I.message.substr(0,250),line:I.lineNumber,level:h,"client.name":Ie.name},postParams:{url:L("PAGE_NAME",window.location.href),file:I.fileName},method:"POST"};Ie.version&&(cc["client.version"]=Ie.version);if(cc.postParams){I.stack&&(cc.postParams.stack=
I.stack);for(var aq=p(Object.keys(Ie)),xk=aq.next();!xk.done;xk=aq.next()){var bq=xk.value;cc.postParams["client."+bq]=Ie[bq]}var yk=L("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS");if(yk)for(var cq=p(Object.keys(yk)),zk=cq.next();!zk.done;zk=cq.next()){var dq=zk.value;cc.postParams[dq]=yk[dq]}var eq=L("SERVER_NAME"),fq=L("SERVER_VERSION");eq&&fq&&(cc.postParams["server.name"]=eq,cc.postParams["server.version"]=fq)}ql(L("ECATCHER_REPORT_HOST","")+"/error_204",cc)}try{cs.add(I.message)}catch(Fx){}ds++}}}}
function ks(){for(var a=p(hs),b=a.next();!b.done;b=a.next())if($l(b.value.toLowerCase()))return!0;return!1}
function ls(a){var b=Ja.apply(1,arguments);a.args||(a.args=[]);a.args.push.apply(a.args,ia(b))}
;function ms(){this.register=new Map}
function ns(a){a=p(a.register.values());for(var b=a.next();!b.done;b=a.next())b.value.Dr("ABORTED")}
ms.prototype.clear=function(){ns(this);this.register.clear()};
var os=new ms;var ps=Date.now().toString();
function qs(){a:{if(window.crypto&&window.crypto.getRandomValues)try{var a=Array(16),b=new Uint8Array(16);window.crypto.getRandomValues(b);for(var c=0;c<a.length;c++)a[c]=b[c];var d=a;break a}catch(e){}d=Array(16);for(a=0;16>a;a++){b=Date.now();for(c=0;c<b%23;c++)d[a]=Math.random();d[a]=Math.floor(256*Math.random())}if(ps)for(a=1,b=0;b<ps.length;b++)d[a%16]=d[a%16]^d[(a-1)%16]/4^ps.charCodeAt(b),a++}a=[];for(b=0;b<d.length;b++)a.push("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".charAt(d[b]&63));
return a.join("")}
;var rs=z.ytLoggingDocDocumentNonce_;rs||(rs=qs(),A("ytLoggingDocDocumentNonce_",rs));var ss=rs;var ts={Zh:0,te:1,Ee:2,Yl:3,bi:4,Bq:5,Om:6,Fo:7,Vn:8,to:9,0:"DEFAULT",1:"CHAT",2:"CONVERSATIONS",3:"MINIPLAYER",4:"DIALOG",5:"VOZ",6:"MUSIC_WATCH_TABS",7:"SHARE",8:"PUSH_NOTIFICATIONS",9:"RICH_GRID_WATCH"};function us(a){this.G=a}
function vs(a){return new us({trackingParams:a})}
us.prototype.getAsJson=function(){var a={};void 0!==this.G.trackingParams?a.trackingParams=this.G.trackingParams:(a.veType=this.G.veType,void 0!==this.G.veCounter&&(a.veCounter=this.G.veCounter),void 0!==this.G.elementIndex&&(a.elementIndex=this.G.elementIndex));void 0!==this.G.dataElement&&(a.dataElement=this.G.dataElement.getAsJson());void 0!==this.G.youtubeData&&(a.youtubeData=this.G.youtubeData);return a};
us.prototype.getAsJspb=function(){var a=new Aj;if(void 0!==this.G.trackingParams){var b=this.G.trackingParams;if(null!=b)if("string"===typeof b)b=b?new jd(b,gd):kd();else if(b.constructor!==jd)if(fd(b))b=b.length?new jd(new Uint8Array(b),gd):kd();else throw Error();D(a,1,b)}else void 0!==this.G.veType&&D(a,2,this.G.veType),void 0!==this.G.veCounter&&D(a,6,this.G.veCounter),void 0!==this.G.elementIndex&&D(a,3,this.G.elementIndex);void 0!==this.G.dataElement&&(b=this.G.dataElement.getAsJspb(),G(a,Aj,
7,b));void 0!==this.G.youtubeData&&G(a,Qi,8,this.G.jspbYoutubeData);return a};
us.prototype.toString=function(){return JSON.stringify(this.getAsJson())};
us.prototype.isClientVe=function(){return!this.G.trackingParams&&!!this.G.veType};function ws(a){a=void 0===a?0:a;return 0===a?"client-screen-nonce":"client-screen-nonce."+a}
function xs(a){a=void 0===a?0:a;return 0===a?"ROOT_VE_TYPE":"ROOT_VE_TYPE."+a}
function ys(a){return L(xs(void 0===a?0:a))}
A("yt_logging_screen.getRootVeType",ys);function zs(a){return(a=ys(void 0===a?0:a))?new us({veType:a,youtubeData:void 0,jspbYoutubeData:void 0}):null}
function As(){var a=L("csn-to-ctt-auth-info");a||(a={},Ik("csn-to-ctt-auth-info",a));return a}
function Bs(a){a=L(ws(void 0===a?0:a));if(!a&&!L("USE_CSN_FALLBACK",!0))return null;a||(a="UNDEFINED_CSN");return a?a:null}
A("yt_logging_screen.getCurrentCsn",Bs);function Cs(a,b,c){var d=As();(c=Bs(c))&&delete d[c];b&&(d[a]=b)}
function Ds(a){return As()[a]}
A("yt_logging_screen.getCttAuthInfo",Ds);
function Es(a,b,c,d){c=void 0===c?0:c;if(a!==L(ws(c))||b!==L(xs(c)))if(Cs(a,d,c),Ik(ws(c),a),Ik(xs(c),b),b=function(){setTimeout(function(){if(a)if(M("web_time_via_jspb")){var e=new Bj;D(e,1,ss);D(e,2,a);M("use_default_heartbeat_client")?Qr(e):Qr(e,void 0,sp)}else e={clientDocumentNonce:ss,clientScreenNonce:a},M("use_default_heartbeat_client")?jm("foregroundHeartbeatScreenAssociated",e):Dr("foregroundHeartbeatScreenAssociated",e,sp)},0)},"requestAnimationFrame"in window)try{window.requestAnimationFrame(b)}catch(e){b()}else b()}
A("yt_logging_screen.setCurrentScreen",Es);var Fs=window.yt&&window.yt.msgs_||window.ytcfg&&window.ytcfg.msgs||{};A("yt.msgs_",Fs);function Gs(a){Dk(Fs,arguments)}
;var Hs={se:3611,Dd:27686,Ed:85013,Fd:23462,Hd:157557,Id:42016,Jd:62407,Kd:26926,Gd:43781,Ld:51236,Md:79148,Nd:50160,Od:77504,ae:153587,be:87907,ce:18630,de:54445,ee:80935,ge:152172,he:105675,ie:150723,je:37521,ke:147285,le:47786,me:98349,ne:123695,oe:6827,pe:29434,qe:7282,re:124448,we:32276,ue:76278,xe:147868,ye:147869,ze:93911,Ae:106531,Be:27259,Ce:27262,De:27263,Fe:21759,Ge:27107,He:62936,Ie:160866,Je:49568,Ke:160789,Le:38408,Me:80637,Ne:68727,Oe:68728,Pe:80353,Qe:80356,Re:74610,Se:45707,Te:83962,
Ue:83970,Ve:46713,We:89711,Xe:74612,Ye:155792,Ze:93265,af:74611,bf:131380,df:128979,ef:139311,ff:128978,cf:131391,gf:105350,jf:139312,kf:134800,hf:131392,mf:113533,nf:93252,pf:99357,rf:94521,sf:114252,tf:113532,uf:94522,qf:94583,vf:88E3,wf:139580,xf:93253,yf:93254,zf:94387,Af:94388,Bf:93255,Cf:97424,lf:72502,Df:110111,Ef:76019,Gf:117092,Hf:117093,Ff:89431,If:110466,Jf:77240,Kf:60508,Lf:148123,Mf:148124,Nf:137401,Of:137402,Pf:137046,Qf:73393,Rf:113534,Sf:92098,Tf:131381,Uf:84517,Vf:83759,Wf:162711,
Xf:162712,Yf:80357,Zf:86113,ag:72598,cg:72733,dg:107349,eg:124275,fg:118203,gg:133275,hg:160157,ig:152569,jg:156651,kg:133274,lg:160159,mg:160158,ng:133272,og:133273,pg:133276,qg:144507,rg:143247,sg:156652,tg:143248,ug:143249,vg:143250,wg:143251,xg:156653,yg:144401,Ag:117431,zg:133797,Bg:153964,Cg:128572,Dg:133405,Eg:117429,Fg:117430,Gg:117432,Hg:120080,Ig:117259,Jg:156655,Kg:156654,Lg:121692,Mg:145656,Ng:156656,Og:145655,Pg:145653,Qg:145654,Rg:145657,Sg:132972,Tg:133051,Ug:133658,Vg:132971,Wg:97615,
Yg:143359,Xg:143356,ah:143361,Zg:143358,dh:143360,bh:143357,eh:142303,fh:143353,gh:143354,hh:144479,ih:143355,jh:31402,lh:133624,mh:146477,nh:133623,oh:133622,kh:133621,ph:84774,rh:160801,qh:95117,sh:150497,th:98930,uh:98931,vh:98932,wh:153320,xh:153321,yh:43347,zh:129889,Ah:149123,Bh:45474,Ch:100352,Dh:84758,Eh:98443,Fh:117985,Gh:74613,Hh:155911,Ih:74614,Jh:64502,Kh:136032,Lh:74615,Mh:74616,Nh:122224,Oh:74617,Ph:77820,Qh:74618,Rh:93278,Sh:93274,Th:93275,Uh:93276,Vh:22110,Wh:29433,Xh:133798,Yh:132295,
ai:120541,ci:82047,di:113550,fi:75836,gi:75837,hi:42352,ii:84512,ji:76065,ki:75989,ri:51879,si:16623,ti:32594,vi:27240,wi:32633,xi:74858,yi:156999,Ai:3945,zi:16989,Bi:45520,Ci:25488,Di:25492,Ei:25494,Fi:55760,Gi:14057,Hi:18451,Ii:57204,Ji:57203,Ki:17897,Li:57205,Mi:18198,Ni:17898,Oi:17909,Pi:43980,Qi:46220,Ri:11721,Si:147994,Ti:49954,Ui:96369,Vi:3854,Wi:151633,Xi:56251,Yi:159108,Zi:25624,aj:152036,rj:16906,sj:99999,tj:68172,uj:27068,vj:47973,wj:72773,xj:26970,yj:26971,zj:96805,Aj:17752,Bj:73233,Cj:109512,
Dj:22256,Ej:14115,Fj:22696,Gj:89278,Hj:89277,Ij:109513,Jj:43278,Kj:43459,Lj:43464,Mj:89279,Nj:43717,Oj:55764,Pj:22255,Qj:147912,Rj:89281,Sj:40963,Tj:43277,Uj:43442,Vj:91824,Wj:120137,Xj:96367,Yj:36850,Zj:72694,ak:37414,bk:36851,dk:124863,ck:121343,ek:73491,fk:54473,gk:166861,hk:43375,ik:46674,jk:143815,kk:139095,lk:144402,mk:149968,nk:149969,pk:32473,qk:72901,rk:72906,sk:50947,tk:50612,uk:50613,vk:50942,wk:84938,xk:84943,yk:84939,zk:84941,Ak:84944,Bk:84940,Ck:84942,Dk:35585,Ek:51926,Fk:79983,Gk:63238,
Hk:18921,Ik:63241,Jk:57893,Kk:41182,Lk:135732,Mk:33424,Nk:22207,Ok:42993,Pk:36229,Qk:22206,Rk:22205,Sk:18993,Tk:19001,Uk:18990,Vk:18991,Wk:18997,Xk:18725,Yk:19003,Zk:36874,al:44763,bl:33427,dl:67793,fl:22182,il:37091,jl:34650,kl:50617,ll:47261,ml:22287,nl:25144,ol:97917,pl:62397,ql:150871,rl:150874,sl:125598,ul:137935,vl:36961,wl:108035,xl:27426,yl:27857,zl:27846,Al:27854,Bl:69692,Cl:61411,Dl:39299,El:38696,Fl:62520,Gl:36382,Hl:108701,Il:50663,Jl:36387,Kl:14908,Ll:37533,Ml:105443,Nl:61635,Ol:62274,
Pl:161670,Ql:133818,Rl:65702,Sl:65703,Tl:65701,Ul:76256,Vl:166382,Wl:37671,Xl:49953,Zl:36216,am:28237,bm:39553,cm:29222,dm:26107,em:38050,fm:26108,hm:120745,gm:26109,im:26110,jm:66881,km:28236,lm:14586,mm:160598,nm:57929,om:74723,pm:44098,qm:44099,tm:23528,um:61699,rm:134104,sm:134103,vm:59149,wm:101951,xm:97346,ym:118051,zm:95102,Am:64882,Bm:119505,Cm:63595,Dm:63349,Em:95101,Fm:75240,Gm:27039,Hm:68823,Im:21537,Jm:83464,Km:75707,Lm:83113,Mm:101952,Nm:101953,Pm:79610,Qm:125755,Rm:24402,Sm:24400,Tm:32925,
Vm:57173,Um:156421,Wm:122502,Xm:145268,Ym:138480,Zm:64423,an:64424,bn:33986,cn:100828,dn:129089,en:21409,jn:135155,kn:135156,ln:135157,mn:135158,nn:158225,pn:135159,qn:135160,rn:167651,sn:135161,un:135162,vn:135163,tn:158226,wn:158227,xn:135164,yn:135165,zn:135166,fn:11070,gn:11074,hn:17880,An:14001,Cn:30709,Dn:30707,En:30711,Fn:30710,Gn:30708,Bn:26984,Hn:146143,In:63648,Jn:63649,Kn:111059,Ln:5754,Mn:20445,Nn:151308,On:151152,Rn:130975,Qn:130976,Pn:167637,Sn:110386,Tn:113746,Un:66557,Wn:17310,Xn:28631,
Yn:21589,Zn:164817,ao:154946,bo:68012,co:162617,eo:60480,fo:138664,ho:141121,jo:164502,ko:31571,lo:141978,mo:150105,no:150106,oo:150107,po:150108,qo:76980,ro:41577,so:45469,uo:38669,vo:13768,wo:13777,xo:141842,yo:62985,zo:4724,Ao:59369,Bo:43927,Co:43928,Do:12924,Eo:100355,Ho:56219,Io:27669,Jo:10337,Go:47896,Ko:122629,Mo:139723,Lo:139722,No:121258,Oo:107598,Po:127991,Qo:96639,Ro:107536,So:130169,To:96661,Uo:145188,Vo:96658,Wo:116646,Xo:159428,Yo:121122,Zo:96660,ap:127738,bp:127083,cp:155281,ep:162959,
fp:163566,gp:147842,hp:104443,ip:96659,jp:147595,kp:106442,lp:162776,mp:134840,np:63667,qp:63668,rp:63669,sp:130686,tp:147036,up:78314,vp:147799,wp:148649,xp:55761,yp:127098,zp:134841,Ap:96368,Bp:67374,Cp:48992,Dp:146176,Ep:49956,Fp:31961,Gp:26388,Hp:23811,Ip:5E4,Jp:126250,Kp:96370,Lp:47355,Mp:47356,Np:37935,Op:45521,Pp:21760,Qp:83769,Rp:49977,Sp:49974,Tp:93497,Up:93498,Vp:34325,Wp:140759,Xp:115803,Yp:123707,Zp:100081,aq:35309,bq:68314,cq:25602,fq:100339,gq:143516,hq:59018,iq:18248,jq:50625,kq:9729,
lq:37168,mq:37169,nq:21667,oq:16749,pq:18635,qq:39305,rq:18046,sq:53969,tq:8213,uq:93926,wq:102852,xq:110099,yq:22678,zq:69076,Aq:137575,Cq:139224,Dq:100856,Eq:154430,Fq:17736,Gq:3832,Hq:147111,Iq:55759,Jq:64031,Pq:93044,Qq:93045,Rq:34388,Sq:17657,Tq:17655,Uq:39579,Vq:39578,Wq:77448,Xq:8196,Yq:11357,Zq:69877,br:8197,dr:156512,er:161613,fr:156509,gr:161612,hr:161614,ir:82039};function Is(){var a=ub(Js),b;return(new Af(function(c,d){a.onSuccess=function(e){jl(e)?c(new Ks(e)):d(new Ls("Request failed, status="+(e&&"status"in e?e.status:-1),"net.badstatus",e))};
a.onError=function(e){d(new Ls("Unknown request error","net.unknown",e))};
a.onTimeout=function(e){d(new Ls("Request timed out","net.timeout",e))};
b=ql("//googleads.g.doubleclick.net/pagead/id",a)})).qb(function(c){c instanceof Hf&&b.abort();
return Ff(c)})}
function Ls(a,b,c){ab.call(this,a+", errorCode="+b);this.errorCode=b;this.xhr=c;this.name="PromiseAjaxError"}
u(Ls,ab);function Ks(a){this.xhr=a}
;function Ms(){this.h=0;this.ba=null}
Ms.prototype.then=function(a,b,c){return 1===this.h&&a?(a=a.call(c,this.ba))&&"function"===typeof a.then?a:Ns(a):2===this.h&&b?(a=b.call(c,this.ba))&&"function"===typeof a.then?a:Os(a):this};
Ms.prototype.getValue=function(){return this.ba};
Ms.prototype.$goog_Thenable=!0;function Os(a){var b=new Ms;a=void 0===a?null:a;b.h=2;b.ba=void 0===a?null:a;return b}
function Ns(a){var b=new Ms;a=void 0===a?null:a;b.h=1;b.ba=void 0===a?null:a;return b}
;function Ps(a,b){var c=void 0===c?{}:c;a={method:void 0===b?"POST":b,mode:el(a)?"same-origin":"cors",credentials:el(a)?"same-origin":"include"};b={};for(var d=p(Object.keys(c)),e=d.next();!e.done;e=d.next())e=e.value,c[e]&&(b[e]=c[e]);0<Object.keys(b).length&&(a.headers=b);return a}
;function Qs(){return vg()||Zl&&$l("applewebkit")&&!$l("version")&&(!$l("safari")||$l("gsa/"))||Mc&&$l("version/")?!0:L("EOM_VISITOR_DATA")?!1:!0}
;function Rs(a){a:{var b=a.raw_embedded_player_response;if(!b&&(a=a.embedded_player_response))try{b=JSON.parse(a)}catch(d){b="EMBEDDED_PLAYER_MODE_UNKNOWN";break a}if(b)b:{for(var c in kj)if(kj[c]==b.embeddedPlayerMode){b=kj[c];break b}b="EMBEDDED_PLAYER_MODE_UNKNOWN"}else b="EMBEDDED_PLAYER_MODE_UNKNOWN"}return"EMBEDDED_PLAYER_MODE_PFL"===b}
;function Ss(a){ab.call(this,a.message||a.description||a.name);this.isMissing=a instanceof Ts;this.isTimeout=a instanceof Ls&&"net.timeout"==a.errorCode;this.isCanceled=a instanceof Hf}
u(Ss,ab);Ss.prototype.name="BiscottiError";function Ts(){ab.call(this,"Biscotti ID is missing from server")}
u(Ts,ab);Ts.prototype.name="BiscottiMissingError";var Js={format:"RAW",method:"GET",timeout:5E3,withCredentials:!0},Us=null;function Vs(){if(M("disable_biscotti_fetch_entirely_for_all_web_clients"))return Error("Biscotti id fetching has been disabled entirely.");if(!Qs())return Error("User has not consented - not fetching biscotti id.");var a=L("PLAYER_VARS",{});if("1"==sb(a))return Error("Biscotti ID is not available in private embed mode");if(Rs(a))return Error("Biscotti id fetching has been disabled for pfl.")}
function Uk(){var a=Vs();if(void 0!==a)return Ff(a);Us||(Us=Is().then(Ws).qb(function(b){return Xs(2,b)}));
return Us}
function Ws(a){a=a.xhr.responseText;if(0!=a.lastIndexOf(")]}'",0))throw new Ts;a=JSON.parse(a.substr(4));if(1<(a.type||1))throw new Ts;a=a.id;Vk(a);Us=Ns(a);Ys(18E5,2);return a}
function Xs(a,b){b=new Ss(b);Vk("");Us=Os(b);0<a&&Ys(12E4,a-1);throw b;}
function Ys(a,b){kl(function(){Is().then(Ws,function(c){return Xs(b,c)}).qb(cb)},a)}
function Zs(){try{var a=B("yt.ads.biscotti.getId_");return a?a():Uk()}catch(b){return Ff(b)}}
;function $s(a){if("1"!=sb(L("PLAYER_VARS",{}))){a&&Tk();try{Zs().then(function(){},function(){}),kl($s,18E5)}catch(b){Rk(b)}}}
;function at(){var a=El.getInstance(),b=Hl(119),c=1<window.devicePixelRatio;if(document.body&&ji(document.body,"exp-invert-logo"))if(c&&!ji(document.body,"inverted-hdpi")){var d=document.body;if(d.classList)d.classList.add("inverted-hdpi");else if(!ji(d,"inverted-hdpi")){var e=hi(d);ii(d,e+(0<e.length?" inverted-hdpi":"inverted-hdpi"))}}else!c&&ji(document.body,"inverted-hdpi")&&ki();if(b!=c){b="f"+(Math.floor(119/31)+1);d=Il(b)||0;d=c?d|67108864:d&-67108865;0==d?delete Dl[b]:(c=d.toString(16),Dl[b]=
c.toString());c=!0;M("web_secure_pref_cookie_killswitch")&&(c=!1);b=a.h;d=[];for(var f in Dl)d.push(f+"="+encodeURIComponent(String(Dl[f])));Al(b,d.join("&"),63072E3,a.i,c)}}
;function bt(){this.wd=!0}
function ct(a){var b={},c=xg([]);c&&(b.Authorization=c,c=a=null==a?void 0:a.sessionIndex,void 0===c&&(c=Number(L("SESSION_INDEX",0)),c=isNaN(c)?0:c),M("voice_search_auth_header_removal")||(b["X-Goog-AuthUser"]=c),"INNERTUBE_HOST_OVERRIDE"in Hk||(b["X-Origin"]=window.location.origin),void 0===a&&"DELEGATED_SESSION_ID"in Hk&&(b["X-Goog-PageId"]=L("DELEGATED_SESSION_ID")));return b}
;var dt={identityType:"UNAUTHENTICATED_IDENTITY_TYPE_UNKNOWN"};var et=new Map([["dark","USER_INTERFACE_THEME_DARK"],["light","USER_INTERFACE_THEME_LIGHT"]]);function ft(){var a=void 0===a?window.location.href:a;if(M("kevlar_disable_theme_param"))return null;lc(mc(5,a));try{var b=cl(a).theme;return et.get(b)||null}catch(c){}return null}
;function gt(){this.h={};if(this.i=Cl()){var a=tg.get("CONSISTENCY",void 0);a&&ht(this,{encryptedTokenJarContents:a})}}
gt.prototype.handleResponse=function(a,b){if(!b)throw Error("request needs to be passed into ConsistencyService");var c,d;b=(null==(c=b.ga.context)?void 0:null==(d=c.request)?void 0:d.consistencyTokenJars)||[];var e;if(a=null==(e=a.responseContext)?void 0:e.consistencyTokenJar){e=p(b);for(c=e.next();!c.done;c=e.next())delete this.h[c.value.encryptedTokenJarContents];ht(this,a)}};
function ht(a,b){if(b.encryptedTokenJarContents&&(a.h[b.encryptedTokenJarContents]=b,"string"===typeof b.expirationSeconds)){var c=Number(b.expirationSeconds);setTimeout(function(){delete a.h[b.encryptedTokenJarContents]},1E3*c);
a.i&&Al("CONSISTENCY",b.encryptedTokenJarContents,c,void 0,!0)}}
;var jt=window.location.hostname.split(".").slice(-2).join(".");function kt(){var a=L("LOCATION_PLAYABILITY_TOKEN");"TVHTML5"===L("INNERTUBE_CLIENT_NAME")&&(this.h=lt(this))&&(a=this.h.get("yt-location-playability-token"));a&&(this.locationPlayabilityToken=a,this.i=void 0)}
var mt;kt.getInstance=function(){mt=B("yt.clientLocationService.instance");mt||(mt=new kt,A("yt.clientLocationService.instance",mt));return mt};
l=kt.prototype;l.setLocationOnInnerTubeContext=function(a){a.client||(a.client={});this.i?(a.client.locationInfo||(a.client.locationInfo={}),a.client.locationInfo.latitudeE7=Math.floor(1E7*this.i.coords.latitude),a.client.locationInfo.longitudeE7=Math.floor(1E7*this.i.coords.longitude),a.client.locationInfo.horizontalAccuracyMeters=Math.round(this.i.coords.accuracy),a.client.locationInfo.forceLocationPlayabilityTokenRefresh=!0):this.locationPlayabilityToken&&(a.client.locationPlayabilityToken=this.locationPlayabilityToken)};
l.handleResponse=function(a){var b;a=null==(b=a.responseContext)?void 0:b.locationPlayabilityToken;void 0!==a&&(this.locationPlayabilityToken=a,this.i=void 0,"TVHTML5"===L("INNERTUBE_CLIENT_NAME")?(this.h=lt(this))&&this.h.set("yt-location-playability-token",a,15552E3):Al("YT_CL",JSON.stringify({loctok:a}),15552E3,jt,!0))};
function lt(a){return void 0===a.h?new am("yt-client-location"):a.h}
l.clearLocationPlayabilityToken=function(a){"TVHTML5"===a?(this.h=lt(this))&&this.h.remove("yt-location-playability-token"):Bl("YT_CL")};
l.getCurrentPositionFromGeolocation=function(){var a=this;if(!(navigator&&navigator.geolocation&&navigator.geolocation.getCurrentPosition))return Promise.reject(Error("Geolocation unsupported"));var b=!1,c=1E4;"MWEB"===L("INNERTUBE_CLIENT_NAME")&&(b=!0,c=15E3);return new Promise(function(d,e){navigator.geolocation.getCurrentPosition(function(f){a.i=f;d(f)},function(f){e(f)},{enableHighAccuracy:b,
maximumAge:0,timeout:c})})};
l.createUnpluggedLocationInfo=function(a){var b={};a=a.coords;if(null==a?0:a.latitude)b.latitudeE7=Math.floor(1E7*a.latitude);if(null==a?0:a.longitude)b.longitudeE7=Math.floor(1E7*a.longitude);if(null==a?0:a.accuracy)b.locationRadiusMeters=Math.round(a.accuracy);return b};function nt(a,b){var c,d=null==(c=Rq(a,jj))?void 0:c.signal;if(d&&b.Qa&&(c=b.Qa[d]))return c();var e;if((c=null==(e=Rq(a,hj))?void 0:e.request)&&b.Lc&&(e=b.Lc[c]))return e();for(var f in a)if(b.Tb[f]&&(a=b.Tb[f]))return a()}
;function ot(a){return function(){return new a}}
;var pt={},qt=(pt.WEB_UNPLUGGED="^unplugged/",pt.WEB_UNPLUGGED_ONBOARDING="^unplugged/",pt.WEB_UNPLUGGED_OPS="^unplugged/",pt.WEB_UNPLUGGED_PUBLIC="^unplugged/",pt.WEB_CREATOR="^creator/",pt.WEB_KIDS="^kids/",pt.WEB_EXPERIMENTS="^experiments/",pt.WEB_MUSIC="^music/",pt.WEB_REMIX="^music/",pt.WEB_MUSIC_EMBEDDED_PLAYER="^music/",pt.WEB_MUSIC_EMBEDDED_PLAYER="^main_app/|^sfv/",pt);
function rt(a){var b=void 0===b?"UNKNOWN_INTERFACE":b;if(1===a.length)return a[0];var c=qt[b];if(c){var d=new RegExp(c),e=p(a);for(c=e.next();!c.done;c=e.next())if(c=c.value,d.exec(c))return c}var f=[];Object.entries(qt).forEach(function(g){var h=p(g);g=h.next().value;h=h.next().value;b!==g&&f.push(h)});
d=new RegExp(f.join("|"));a.sort(function(g,h){return g.length-h.length});
e=p(a);for(c=e.next();!c.done;c=e.next())if(c=c.value,!d.exec(c))return c;return a[0]}
;function st(){}
st.prototype.m=function(a,b,c){b=void 0===b?{}:b;c=void 0===c?dt:c;var d=a.clickTrackingParams,e=this.l,f=!1;f=void 0===f?!1:f;e=void 0===e?!1:e;var g=L("INNERTUBE_CONTEXT");if(g){g=vb(g);M("web_no_tracking_params_in_shell_killswitch")||delete g.clickTracking;g.client||(g.client={});var h=g.client;"MWEB"===h.clientName&&(h.clientFormFactor=L("IS_TABLET")?"LARGE_FORM_FACTOR":"SMALL_FORM_FACTOR");h.screenWidthPoints=window.innerWidth;h.screenHeightPoints=window.innerHeight;h.screenPixelDensity=Math.round(window.devicePixelRatio||
1);h.screenDensityFloat=window.devicePixelRatio||1;h.utcOffsetMinutes=-Math.floor((new Date).getTimezoneOffset());var k=void 0===k?!1:k;El.getInstance();var m="USER_INTERFACE_THEME_LIGHT";Hl(165)?m="USER_INTERFACE_THEME_DARK":Hl(174)?m="USER_INTERFACE_THEME_LIGHT":!M("kevlar_legacy_browsers")&&window.matchMedia&&window.matchMedia("(prefers-color-scheme)").matches&&window.matchMedia("(prefers-color-scheme: dark)").matches&&(m="USER_INTERFACE_THEME_DARK");k=k?m:ft()||m;h.userInterfaceTheme=k;if(!f){if(k=
Ol())h.connectionType=k;M("web_log_effective_connection_type")&&(k=Pl())&&(g.client.effectiveConnectionType=k)}var q;if(M("web_log_memory_total_kbytes")&&(null==(q=z.navigator)?0:q.deviceMemory)){var r;q=null==(r=z.navigator)?void 0:r.deviceMemory;g.client.memoryTotalKbytes=""+1E6*q}r=cl(z.location.href);!M("web_populate_internal_geo_killswitch")&&r.internalcountrycode&&(h.internalGeo=r.internalcountrycode);"MWEB"===h.clientName||"WEB"===h.clientName?(h.mainAppWebInfo={graftUrl:z.location.href},M("kevlar_woffle")&&
yl.h&&(r=yl.h,h.mainAppWebInfo.pwaInstallabilityStatus=!r.h&&r.i?"PWA_INSTALLABILITY_STATUS_CAN_BE_INSTALLED":"PWA_INSTALLABILITY_STATUS_UNKNOWN"),h.mainAppWebInfo.webDisplayMode=zl(),h.mainAppWebInfo.isWebNativeShareAvailable=navigator&&void 0!==navigator.share):"TVHTML5"===h.clientName&&(!M("web_lr_app_quality_killswitch")&&(r=L("LIVING_ROOM_APP_QUALITY"))&&(h.tvAppInfo=Object.assign(h.tvAppInfo||{},{appQuality:r})),r=L("LIVING_ROOM_CERTIFICATION_SCOPE"))&&(h.tvAppInfo=Object.assign(h.tvAppInfo||
{},{certificationScope:r}));if(!M("web_populate_time_zone_itc_killswitch")){b:{if("undefined"!==typeof Intl)try{var w=(new Intl.DateTimeFormat).resolvedOptions().timeZone;break b}catch(rb){}w=void 0}w&&(h.timeZone=w)}(w=Mk())?h.experimentsToken=w:delete h.experimentsToken;w=Nk();gt.h||(gt.h=new gt);h=gt.h.h;r=[];q=0;for(var t in h)r[q++]=h[t];g.request=Object.assign({},g.request,{internalExperimentFlags:w,consistencyTokenJars:r});!M("web_prequest_context_killswitch")&&(t=L("INNERTUBE_CONTEXT_PREQUEST_CONTEXT"))&&
(g.request.externalPrequestContext=t);w=El.getInstance();t=Hl(58);w=w.get("gsml","");g.user=Object.assign({},g.user);t&&(g.user.enableSafetyMode=t);w&&(g.user.lockedSafetyMode=!0);M("warm_op_csn_cleanup")?e&&(f=Bs())&&(g.clientScreenNonce=f):!f&&(f=Bs())&&(g.clientScreenNonce=f);d&&(g.clickTracking={clickTrackingParams:d});if(d=B("yt.mdx.remote.remoteClient_"))g.remoteClient=d;kt.getInstance().setLocationOnInnerTubeContext(g);try{var y=fl(),E=y.bid;delete y.bid;g.adSignalsInfo={params:[],bid:E};var F=
p(Object.entries(y));for(var O=F.next();!O.done;O=F.next()){var N=p(O.value),R=N.next().value,ca=N.next().value;y=R;E=ca;d=void 0;null==(d=g.adSignalsInfo.params)||d.push({key:y,value:""+E})}}catch(rb){is(rb)}F=g}else is(Error("Error: No InnerTubeContext shell provided in ytconfig.")),F={};F={context:F};if(O=this.h(a)){this.i(F,O,b);var U;b="/youtubei/v1/"+rt(this.j());(O=null==(U=Rq(a.commandMetadata,ij))?void 0:U.apiUrl)&&(b=O);U=b;(b=L("INNERTUBE_HOST_OVERRIDE"))&&(U=String(b)+String(oc(U)));b=
{};b.key=L("INNERTUBE_API_KEY");M("json_condensed_response")&&(b.prettyPrint="false");U=dl(U,b||{},!1);a=Object.assign({},{command:a},void 0);a={input:U,xa:Ps(U),ga:F,config:a};a.config.ab?a.config.ab.identity=c:a.config.ab={identity:c};return a}is(new P("Error: Failed to create Request from Command.",a))};
fa.Object.defineProperties(st.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return!1}}});function tt(){}
u(tt,st);tt.prototype.m=function(){return{input:"/getDatasyncIdsEndpoint",xa:Ps("/getDatasyncIdsEndpoint","GET"),ga:{}}};
tt.prototype.j=function(){return[]};
tt.prototype.h=function(){};
tt.prototype.i=function(){};var ut={},vt=(ut.GET_DATASYNC_IDS=ot(tt),ut);function wt(a){var b=Ja.apply(1,arguments);if(!xt(a)||b.some(function(d){return!xt(d)}))throw Error("Only objects may be merged.");
b=p(b);for(var c=b.next();!c.done;c=b.next())zt(a,c.value);return a}
function zt(a,b){for(var c in b)if(xt(b[c])){if(c in a&&!xt(a[c]))throw Error("Cannot merge an object into a non-object.");c in a||(a[c]={});zt(a[c],b[c])}else if(At(b[c])){if(c in a&&!At(a[c]))throw Error("Cannot merge an array into a non-array.");c in a||(a[c]=[]);Bt(a[c],b[c])}else a[c]=b[c];return a}
function Bt(a,b){b=p(b);for(var c=b.next();!c.done;c=b.next())c=c.value,xt(c)?a.push(zt({},c)):At(c)?a.push(Bt([],c)):a.push(c);return a}
function xt(a){return"object"===typeof a&&!Array.isArray(a)}
function At(a){return"object"===typeof a&&Array.isArray(a)}
;function Ct(a,b){uo.call(this,1,arguments);this.timer=b}
u(Ct,uo);var Dt=new vo("aft-recorded",Ct);var Et=window;function Ft(){this.timing={};this.clearResourceTimings=function(){};
this.webkitClearResourceTimings=function(){};
this.mozClearResourceTimings=function(){};
this.msClearResourceTimings=function(){};
this.oClearResourceTimings=function(){}}
var S=Et.performance||Et.mozPerformance||Et.msPerformance||Et.webkitPerformance||new Ft;var Gt=!1,Ht={'script[name="scheduler/scheduler"]':"sj",'script[name="player/base"]':"pj",'link[rel="stylesheet"][name="www-player"]':"pc",'link[rel="stylesheet"][name="player/www-player"]':"pc",'script[name="desktop_polymer/desktop_polymer"]':"dpj",'link[rel="import"][name="desktop_polymer"]':"dph",'script[name="mobile-c3"]':"mcj",'link[rel="stylesheet"][name="mobile-c3"]':"mcc",'script[name="player-plasma-ias-phone/base"]':"mcppj",'script[name="player-plasma-ias-tablet/base"]':"mcptj",'link[rel="stylesheet"][name="mobile-polymer-player-ias"]':"mcpc",
'link[rel="stylesheet"][name="mobile-polymer-player-svg-ias"]':"mcpsc",'script[name="mobile_blazer_core_mod"]':"mbcj",'link[rel="stylesheet"][name="mobile_blazer_css"]':"mbc",'script[name="mobile_blazer_logged_in_users_mod"]':"mbliuj",'script[name="mobile_blazer_logged_out_users_mod"]':"mblouj",'script[name="mobile_blazer_noncore_mod"]':"mbnj","#player_css":"mbpc",'script[name="mobile_blazer_desktopplayer_mod"]':"mbpj",'link[rel="stylesheet"][name="mobile_blazer_tablet_css"]':"mbtc",'script[name="mobile_blazer_watch_mod"]':"mbwj"};
Xa(S.clearResourceTimings||S.webkitClearResourceTimings||S.mozClearResourceTimings||S.msClearResourceTimings||S.oClearResourceTimings||cb,S);function It(a){var b=Jt("aft",a);if(b)return b;b=L((a||"")+"TIMING_AFT_KEYS",["ol"]);for(var c=b.length,d=0;d<c;d++){var e=Jt(b[d],a);if(e)return e}return NaN}
function Kt(){var a;if(M("csi_use_performance_navigation_timing")||M("csi_use_performance_navigation_timing_tvhtml5")){var b,c,d,e=null==S?void 0:null==(a=S.getEntriesByType)?void 0:null==(b=a.call(S,"navigation"))?void 0:null==(c=b[0])?void 0:null==(d=c.toJSON)?void 0:d.call(c);e?(e.requestStart=Lt(e.requestStart),e.responseEnd=Lt(e.responseEnd),e.redirectStart=Lt(e.redirectStart),e.redirectEnd=Lt(e.redirectEnd),e.domainLookupEnd=Lt(e.domainLookupEnd),e.connectStart=Lt(e.connectStart),e.connectEnd=
Lt(e.connectEnd),e.responseStart=Lt(e.responseStart),e.secureConnectionStart=Lt(e.secureConnectionStart),e.domainLookupStart=Lt(e.domainLookupStart),e.isPerformanceNavigationTiming=!0,a=e):a=S.timing}else a=S.timing;return a}
function Mt(){return(M("csi_use_time_origin")||M("csi_use_time_origin_tvhtml5"))&&S.timeOrigin?Math.floor(S.timeOrigin):S.timing.navigationStart}
function Lt(a){return Math.round(Mt()+a)}
function Nt(a){var b;(b=B("ytcsi."+(a||"")+"data_"))||(b={tick:{},info:{}},A("ytcsi."+(a||"")+"data_",b));return b}
function Ot(a){a=Nt(a);a.info||(a.info={});return a.info}
function Pt(a){a=Nt(a);a.metadata||(a.metadata={});return a.metadata}
function Qt(a){a=Nt(a);a.tick||(a.tick={});return a.tick}
function Jt(a,b){if(a=Qt(b)[a])return"number"===typeof a?a:a[a.length-1]}
function Rt(a){a=Nt(a);if(a.gel){var b=a.gel;b.gelInfos||(b.gelInfos={});b.gelTicks||(b.gelTicks={})}else a.gel={gelTicks:{},gelInfos:{}};return a.gel}
function St(a){a=Rt(a);a.gelInfos||(a.gelInfos={});return a.gelInfos}
function Tt(a){var b=Nt(a).nonce;b||(b=qs(),Nt(a).nonce=b);return b}
function Ut(a){var b=Jt("_start",a),c=It(a);b&&c&&!Gt&&(Ao(Dt,new Ct(Math.round(c-b),a)),Gt=!0)}
function Vt(a,b){for(var c=p(Object.keys(b)),d=c.next();!d.done;d=c.next())if(d=d.value,!Object.keys(a).includes(d)||"object"===typeof b[d]&&!Vt(a[d],b[d]))return!1;return!0}
;function Wt(){if(S.getEntriesByType){var a=S.getEntriesByType("paint");if(a=jb(a,function(b){return"first-paint"===b.name}))return Lt(a.startTime)}a=S.timing;
return a.bd?Math.max(0,a.bd):0}
;function Xt(){var a=B("ytcsi.debug");a||(a=[],A("ytcsi.debug",a),A("ytcsi.reference",{}));return a}
function Yt(a){a=a||"";var b=B("ytcsi.reference");b||(Xt(),b=B("ytcsi.reference"));if(b[a])return b[a];var c=Xt(),d={timerName:a,info:{},tick:{},span:{},jspbInfo:[]};c.push(d);return b[a]=d}
;var T={},Zt=(T.auto_search="LATENCY_ACTION_AUTO_SEARCH",T.ad_to_ad="LATENCY_ACTION_AD_TO_AD",T.ad_to_video="LATENCY_ACTION_AD_TO_VIDEO",T["analytics.explore"]="LATENCY_ACTION_CREATOR_ANALYTICS_EXPLORE",T.app_startup="LATENCY_ACTION_APP_STARTUP",T["artist.analytics"]="LATENCY_ACTION_CREATOR_ARTIST_ANALYTICS",T["artist.events"]="LATENCY_ACTION_CREATOR_ARTIST_CONCERTS",T["artist.presskit"]="LATENCY_ACTION_CREATOR_ARTIST_PROFILE",T["song.analytics"]="LATENCY_ACTION_CREATOR_SONG_ANALYTICS",T.browse="LATENCY_ACTION_BROWSE",
T.cast_splash="LATENCY_ACTION_CAST_SPLASH",T.channels="LATENCY_ACTION_CHANNELS",T.creator_channel_dashboard="LATENCY_ACTION_CREATOR_CHANNEL_DASHBOARD",T["channel.analytics"]="LATENCY_ACTION_CREATOR_CHANNEL_ANALYTICS",T["channel.comments"]="LATENCY_ACTION_CREATOR_CHANNEL_COMMENTS",T["channel.content"]="LATENCY_ACTION_CREATOR_POST_LIST",T["channel.content.promotions"]="LATENCY_ACTION_CREATOR_PROMOTION_LIST",T["channel.copyright"]="LATENCY_ACTION_CREATOR_CHANNEL_COPYRIGHT",T["channel.editing"]="LATENCY_ACTION_CREATOR_CHANNEL_EDITING",
T["channel.monetization"]="LATENCY_ACTION_CREATOR_CHANNEL_MONETIZATION",T["channel.music"]="LATENCY_ACTION_CREATOR_CHANNEL_MUSIC",T["channel.music_storefront"]="LATENCY_ACTION_CREATOR_CHANNEL_MUSIC_STOREFRONT",T["channel.playlists"]="LATENCY_ACTION_CREATOR_CHANNEL_PLAYLISTS",T["channel.translations"]="LATENCY_ACTION_CREATOR_CHANNEL_TRANSLATIONS",T["channel.videos"]="LATENCY_ACTION_CREATOR_CHANNEL_VIDEOS",T["channel.live_streaming"]="LATENCY_ACTION_CREATOR_LIVE_STREAMING",T.chips="LATENCY_ACTION_CHIPS",
T["dialog.copyright_strikes"]="LATENCY_ACTION_CREATOR_DIALOG_COPYRIGHT_STRIKES",T["dialog.video_copyright"]="LATENCY_ACTION_CREATOR_DIALOG_VIDEO_COPYRIGHT",T["dialog.uploads"]="LATENCY_ACTION_CREATOR_DIALOG_UPLOADS",T.direct_playback="LATENCY_ACTION_DIRECT_PLAYBACK",T.embed="LATENCY_ACTION_EMBED",T.entity_key_serialization_perf="LATENCY_ACTION_ENTITY_KEY_SERIALIZATION_PERF",T.entity_key_deserialization_perf="LATENCY_ACTION_ENTITY_KEY_DESERIALIZATION_PERF",T.explore="LATENCY_ACTION_EXPLORE",T.home=
"LATENCY_ACTION_HOME",T.library="LATENCY_ACTION_LIBRARY",T.live="LATENCY_ACTION_LIVE",T.live_pagination="LATENCY_ACTION_LIVE_PAGINATION",T.onboarding="LATENCY_ACTION_ONBOARDING",T["owner.issues"]="LATENCY_ACTION_CREATOR_CMS_ISSUES",T.parent_profile_settings="LATENCY_ACTION_KIDS_PARENT_PROFILE_SETTINGS",T.parent_tools_collection="LATENCY_ACTION_PARENT_TOOLS_COLLECTION",T.parent_tools_dashboard="LATENCY_ACTION_PARENT_TOOLS_DASHBOARD",T.player_att="LATENCY_ACTION_PLAYER_ATTESTATION",T["post.comments"]=
"LATENCY_ACTION_CREATOR_POST_COMMENTS",T["post.edit"]="LATENCY_ACTION_CREATOR_POST_EDIT",T.prebuffer="LATENCY_ACTION_PREBUFFER",T.prefetch="LATENCY_ACTION_PREFETCH",T.profile_settings="LATENCY_ACTION_KIDS_PROFILE_SETTINGS",T.profile_switcher="LATENCY_ACTION_LOGIN",T.reel_watch="LATENCY_ACTION_REEL_WATCH",T.results="LATENCY_ACTION_RESULTS",T["promotion.edit"]="LATENCY_ACTION_CREATOR_PROMOTION_EDIT",T.search_ui="LATENCY_ACTION_SEARCH_UI",T.search_suggest="LATENCY_ACTION_SUGGEST",T.search_zero_state=
"LATENCY_ACTION_SEARCH_ZERO_STATE",T.secret_code="LATENCY_ACTION_KIDS_SECRET_CODE",T.seek="LATENCY_ACTION_PLAYER_SEEK",T.settings="LATENCY_ACTION_SETTINGS",T.store="LATENCY_ACTION_STORE",T.tenx="LATENCY_ACTION_TENX",T.video_to_ad="LATENCY_ACTION_VIDEO_TO_AD",T.watch="LATENCY_ACTION_WATCH",T.watch_it_again="LATENCY_ACTION_KIDS_WATCH_IT_AGAIN",T["watch,watch7"]="LATENCY_ACTION_WATCH",T["watch,watch7_html5"]="LATENCY_ACTION_WATCH",T["watch,watch7ad"]="LATENCY_ACTION_WATCH",T["watch,watch7ad_html5"]=
"LATENCY_ACTION_WATCH",T.wn_comments="LATENCY_ACTION_LOAD_COMMENTS",T.ww_rqs="LATENCY_ACTION_WHO_IS_WATCHING",T["video.analytics"]="LATENCY_ACTION_CREATOR_VIDEO_ANALYTICS",T["video.comments"]="LATENCY_ACTION_CREATOR_VIDEO_COMMENTS",T["video.edit"]="LATENCY_ACTION_CREATOR_VIDEO_EDIT",T["video.editor"]="LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR",T["video.editor_async"]="LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR_ASYNC",T["video.live_settings"]="LATENCY_ACTION_CREATOR_VIDEO_LIVE_SETTINGS",T["video.live_streaming"]=
"LATENCY_ACTION_CREATOR_VIDEO_LIVE_STREAMING",T["video.monetization"]="LATENCY_ACTION_CREATOR_VIDEO_MONETIZATION",T["video.translations"]="LATENCY_ACTION_CREATOR_VIDEO_TRANSLATIONS",T.voice_assistant="LATENCY_ACTION_VOICE_ASSISTANT",T.cast_load_by_entity_to_watch="LATENCY_ACTION_CAST_LOAD_BY_ENTITY_TO_WATCH",T.networkless_performance="LATENCY_ACTION_NETWORKLESS_PERFORMANCE",T),V={},$t=(V.ad_allowed="adTypesAllowed",V.yt_abt="adBreakType",V.ad_cpn="adClientPlaybackNonce",V.ad_docid="adVideoId",V.yt_ad_an=
"adNetworks",V.ad_at="adType",V.aida="appInstallDataAgeMs",V.browse_id="browseId",V.p="httpProtocol",V.t="transportProtocol",V.cs="commandSource",V.cpn="clientPlaybackNonce",V.ccs="creatorInfo.creatorCanaryState",V.ctop="creatorInfo.topEntityType",V.csn="clientScreenNonce",V.docid="videoId",V.GetHome_rid="requestIds",V.GetSearch_rid="requestIds",V.GetPlayer_rid="requestIds",V.GetWatchNext_rid="requestIds",V.GetBrowse_rid="requestIds",V.GetLibrary_rid="requestIds",V.is_continuation="isContinuation",
V.is_nav="isNavigation",V.b_p="kabukiInfo.browseParams",V.is_prefetch="kabukiInfo.isPrefetch",V.is_secondary_nav="kabukiInfo.isSecondaryNav",V.nav_type="kabukiInfo.navigationType",V.prev_browse_id="kabukiInfo.prevBrowseId",V.query_source="kabukiInfo.querySource",V.voz_type="kabukiInfo.vozType",V.yt_lt="loadType",V.mver="creatorInfo.measurementVersion",V.yt_ad="isMonetized",V.nr="webInfo.navigationReason",V.nrsu="navigationRequestedSameUrl",V.pnt="performanceNavigationTiming",V.prt="playbackRequiresTap",
V.plt="playerInfo.playbackType",V.pis="playerInfo.playerInitializedState",V.paused="playerInfo.isPausedOnLoad",V.yt_pt="playerType",V.fmt="playerInfo.itag",V.yt_pl="watchInfo.isPlaylist",V.yt_pre="playerInfo.preloadType",V.yt_ad_pr="prerollAllowed",V.pa="previousAction",V.yt_red="isRedSubscriber",V.rce="mwebInfo.responseContentEncoding",V.rc="resourceInfo.resourceCache",V.scrh="screenHeight",V.scrw="screenWidth",V.st="serverTimeMs",V.ssdm="shellStartupDurationMs",V.br_trs="tvInfo.bedrockTriggerState",
V.kebqat="kabukiInfo.earlyBrowseRequestInfo.abandonmentType",V.kebqa="kabukiInfo.earlyBrowseRequestInfo.adopted",V.label="tvInfo.label",V.is_mdx="tvInfo.isMdx",V.preloaded="tvInfo.isPreloaded",V.aac_type="tvInfo.authAccessCredentialType",V.upg_player_vis="playerInfo.visibilityState",V.query="unpluggedInfo.query",V.upg_chip_ids_string="unpluggedInfo.upgChipIdsString",V.yt_vst="videoStreamType",V.vph="viewportHeight",V.vpw="viewportWidth",V.yt_vis="isVisible",V.rcl="mwebInfo.responseContentLength",
V.GetSettings_rid="requestIds",V.GetTrending_rid="requestIds",V.GetMusicSearchSuggestions_rid="requestIds",V.REQUEST_ID="requestIds",V),au="isContinuation isNavigation kabukiInfo.earlyBrowseRequestInfo.adopted kabukiInfo.isPrefetch kabukiInfo.isSecondaryNav isMonetized navigationRequestedSameUrl performanceNavigationTiming playerInfo.isPausedOnLoad prerollAllowed isRedSubscriber tvInfo.isMdx tvInfo.isPreloaded isVisible watchInfo.isPlaylist playbackRequiresTap".split(" "),bu={},cu=(bu.ccs="CANARY_STATE_",
bu.mver="MEASUREMENT_VERSION_",bu.pis="PLAYER_INITIALIZED_STATE_",bu.yt_pt="LATENCY_PLAYER_",bu.pa="LATENCY_ACTION_",bu.ctop="TOP_ENTITY_TYPE_",bu.yt_vst="VIDEO_STREAM_TYPE_",bu),du="all_vc ap aq c cbr cbrand cbrver cmodel cos cosver cplatform ctheme cver ei l_an l_mm plid srt yt_fss yt_li vpst vpni2 vpil2 icrc icrt pa GetAccountOverview_rid GetHistory_rid cmt d_vpct d_vpnfi d_vpni nsru pc pfa pfeh pftr pnc prerender psc rc start tcrt tcrc ssr vpr vps yt_abt yt_fn yt_fs yt_pft yt_pre yt_pt yt_pvis ytu_pvis yt_ref yt_sts tds".split(" ");
function eu(a){return Zt[a]||"LATENCY_ACTION_UNKNOWN"}
function fu(a,b,c){c=Rt(c);if(c.gelInfos)c.gelInfos[a]=!0;else{var d={};c.gelInfos=(d[a]=!0,d)}if(a.match("_rid")){var e=a.split("_rid")[0];a="REQUEST_ID"}if(a in $t){c=$t[a];0<=eb(au,c)&&(b=!!b);a in cu&&"string"===typeof b&&(b=cu[a]+b.toUpperCase());a=b;b=c.split(".");for(var f=d={},g=0;g<b.length-1;g++){var h=b[g];f[h]={};f=f[h]}f[b[b.length-1]]="requestIds"===c?[{id:a,endpoint:e}]:a;return wt({},d)}0<=eb(du,a)||js(new P("Unknown label logged with GEL CSI",a))}
;var W={LATENCY_ACTION_SHORTS_VIDEO_INGESTION_TRANSCODING:179,LATENCY_ACTION_KIDS_PROFILE_SWITCHER:90,LATENCY_ACTION_OFFLINE_THUMBNAIL_TRANSFER:100,LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR_ASYNC:46,LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR:37,LATENCY_ACTION_SPINNER_DISPLAYED:14,LATENCY_ACTION_PLAYABILITY_CHECK:10,LATENCY_ACTION_PROCESS:9,LATENCY_ACTION_APP_STARTUP:5,LATENCY_ACTION_COMMERCE_TRANSACTION:184,LATENCY_ACTION_LOG_PAYMENT_SERVER_ANALYTICS_RPC:173,LATENCY_ACTION_GET_PAYMENT_INSTRUMENTS_PARAMS_RPC:172,
LATENCY_ACTION_GET_FIX_INSTRUMENT_PARAMS_RPC:171,LATENCY_ACTION_RESUME_SUBSCRIPTION_RPC:170,LATENCY_ACTION_PAUSE_SUBSCRIPTION_RPC:169,LATENCY_ACTION_GET_OFFLINE_UPSELL_RPC:168,LATENCY_ACTION_GET_OFFERS_RPC:167,LATENCY_ACTION_GET_CANCELLATION_YT_FLOW_RPC:166,LATENCY_ACTION_GET_CANCELLATION_FLOW_RPC:165,LATENCY_ACTION_UPDATE_CROSS_DEVICE_OFFLINE_STATE_RPC:164,LATENCY_ACTION_GET_OFFER_DETAILS_RPC:163,LATENCY_ACTION_CANCEL_RECURRENCE_TRANSACTION_RPC:162,LATENCY_ACTION_GET_TIP_MODULE_RPC:161,LATENCY_ACTION_HANDLE_TRANSACTION_RPC:160,
LATENCY_ACTION_COMPLETE_TRANSACTION_RPC:159,LATENCY_ACTION_GET_CART_RPC:158,LATENCY_ACTION_THUMBNAIL_FETCH:156,LATENCY_ACTION_ABANDONED_DIRECT_PLAYBACK:154,LATENCY_ACTION_SHARE_VIDEO:153,LATENCY_ACTION_AD_TO_VIDEO_INT:152,LATENCY_ACTION_ABANDONED_BROWSE:151,LATENCY_ACTION_PLAYER_ROTATION:150,LATENCY_ACTION_GENERIC_WEB_VIEW:183,LATENCY_ACTION_SHOPPING_IN_APP:124,LATENCY_ACTION_PLAYER_ATTESTATION:121,LATENCY_ACTION_PLAYER_SEEK:119,LATENCY_ACTION_SUPER_STICKER_BUY_FLOW:114,LATENCY_ACTION_DOWNLOADS_DATA_ACCESS:180,
LATENCY_ACTION_BLOCKS_PERFORMANCE:148,LATENCY_ACTION_ASSISTANT_QUERY:138,LATENCY_ACTION_ASSISTANT_SETTINGS:137,LATENCY_ACTION_ENTITY_KEY_DESERIALIZATION_PERF:129,LATENCY_ACTION_ENTITY_KEY_SERIALIZATION_PERF:128,LATENCY_ACTION_PROOF_OF_ORIGIN_TOKEN_CREATE:127,LATENCY_ACTION_EMBEDS_SDK_INITIALIZATION:123,LATENCY_ACTION_NETWORKLESS_PERFORMANCE:122,LATENCY_ACTION_DOWNLOADS_EXPANSION:133,LATENCY_ACTION_ENTITY_TRANSFORM:131,LATENCY_ACTION_DOWNLOADS_COMPATIBILITY_LAYER:96,LATENCY_ACTION_EMBEDS_SET_VIDEO:95,
LATENCY_ACTION_SETTINGS:93,LATENCY_ACTION_ABANDONED_STARTUP:81,LATENCY_ACTION_MEDIA_BROWSER_ALARM_PLAY:80,LATENCY_ACTION_MEDIA_BROWSER_SEARCH:79,LATENCY_ACTION_MEDIA_BROWSER_LOAD_TREE:78,LATENCY_ACTION_WHO_IS_WATCHING:77,LATENCY_ACTION_CAST_LOAD_BY_ENTITY_TO_WATCH:76,LATENCY_ACTION_LITE_SWITCH_ACCOUNT:73,LATENCY_ACTION_ELEMENTS_PERFORMANCE:70,LATENCY_ACTION_LOCATION_SIGNAL_COLLECTION:69,LATENCY_ACTION_MODIFY_CHANNEL_NOTIFICATION:65,LATENCY_ACTION_OFFLINE_STORE_START:61,LATENCY_ACTION_REEL_EDITOR:58,
LATENCY_ACTION_CHANNEL_SUBSCRIBE:56,LATENCY_ACTION_CHANNEL_PREVIEW:55,LATENCY_ACTION_PREFETCH:52,LATENCY_ACTION_ABANDONED_WATCH:45,LATENCY_ACTION_LOAD_COMMENT_REPLIES:26,LATENCY_ACTION_LOAD_COMMENTS:25,LATENCY_ACTION_EDIT_COMMENT:24,LATENCY_ACTION_NEW_COMMENT:23,LATENCY_ACTION_OFFLINE_SHARING_RECEIVER_PAIRING:19,LATENCY_ACTION_EMBED:18,LATENCY_ACTION_MDX_LAUNCH:15,LATENCY_ACTION_RESOLVE_URL:13,LATENCY_ACTION_CAST_SPLASH:149,LATENCY_ACTION_MDX_CONNECT_TO_SESSION:190,LATENCY_ACTION_MDX_STREAM_TRANSFER:178,
LATENCY_ACTION_MDX_CAST:120,LATENCY_ACTION_MDX_COMMAND:12,LATENCY_ACTION_REEL_SELECT_SEGMENT:136,LATENCY_ACTION_ACCELERATED_EFFECTS:145,LATENCY_ACTION_EDIT_AUDIO_GEN:182,LATENCY_ACTION_UPLOAD_AUDIO_MIXER:147,LATENCY_ACTION_SHORTS_CLIENT_SIDE_RENDERING:157,LATENCY_ACTION_SHORTS_SEG_IMP_TRANSCODING:146,LATENCY_ACTION_SHORTS_AUDIO_PICKER_PLAYBACK:130,LATENCY_ACTION_SHORTS_WAVEFORM_DOWNLOAD:125,LATENCY_ACTION_SHORTS_VIDEO_INGESTION:155,LATENCY_ACTION_SHORTS_GALLERY:107,LATENCY_ACTION_SHORTS_TRIM:105,
LATENCY_ACTION_SHORTS_EDIT:104,LATENCY_ACTION_SHORTS_CAMERA:103,LATENCY_ACTION_PRODUCER_EXPORT_PROJECT:193,LATENCY_ACTION_PRODUCER_EDITOR:192,LATENCY_ACTION_PARENT_TOOLS_DASHBOARD:102,LATENCY_ACTION_PARENT_TOOLS_COLLECTION:101,LATENCY_ACTION_MUSIC_LOAD_RECOMMENDED_MEDIA_ITEMS:116,LATENCY_ACTION_MUSIC_LOAD_MEDIA_ITEMS:115,LATENCY_ACTION_MUSIC_ALBUM_DETAIL:72,LATENCY_ACTION_MUSIC_PLAYLIST_DETAIL:71,LATENCY_ACTION_STORE:175,LATENCY_ACTION_CHIPS:68,LATENCY_ACTION_SEARCH_ZERO_STATE:67,LATENCY_ACTION_LIVE_PAGINATION:117,
LATENCY_ACTION_LIVE:20,LATENCY_ACTION_PREBUFFER:40,LATENCY_ACTION_TENX:39,LATENCY_ACTION_KIDS_PROFILE_SETTINGS:94,LATENCY_ACTION_KIDS_WATCH_IT_AGAIN:92,LATENCY_ACTION_KIDS_SECRET_CODE:91,LATENCY_ACTION_KIDS_PARENT_PROFILE_SETTINGS:89,LATENCY_ACTION_KIDS_ONBOARDING:88,LATENCY_ACTION_KIDS_VOICE_SEARCH:82,LATENCY_ACTION_KIDS_CURATED_COLLECTION:62,LATENCY_ACTION_KIDS_LIBRARY:53,LATENCY_ACTION_CREATOR_PROMOTION_LIST:186,LATENCY_ACTION_CREATOR_PROMOTION_EDIT:185,LATENCY_ACTION_CREATOR_VIDEO_TRANSLATIONS:38,
LATENCY_ACTION_CREATOR_VIDEO_MONETIZATION:74,LATENCY_ACTION_CREATOR_VIDEO_LIVE_STREAMING:141,LATENCY_ACTION_CREATOR_VIDEO_LIVE_SETTINGS:142,LATENCY_ACTION_CREATOR_VIDEO_EDITOR_ASYNC:51,LATENCY_ACTION_CREATOR_VIDEO_EDITOR:50,LATENCY_ACTION_CREATOR_VIDEO_EDIT:36,LATENCY_ACTION_CREATOR_VIDEO_COMMENTS:34,LATENCY_ACTION_CREATOR_VIDEO_ANALYTICS:33,LATENCY_ACTION_CREATOR_SONG_ANALYTICS:176,LATENCY_ACTION_CREATOR_POST_LIST:112,LATENCY_ACTION_CREATOR_POST_EDIT:110,LATENCY_ACTION_CREATOR_POST_COMMENTS:111,
LATENCY_ACTION_CREATOR_LIVE_STREAMING:108,LATENCY_ACTION_CREATOR_DIALOG_VIDEO_COPYRIGHT:174,LATENCY_ACTION_CREATOR_DIALOG_UPLOADS:86,LATENCY_ACTION_CREATOR_DIALOG_COPYRIGHT_STRIKES:87,LATENCY_ACTION_CREATOR_CHANNEL_VIDEOS:32,LATENCY_ACTION_CREATOR_CHANNEL_TRANSLATIONS:48,LATENCY_ACTION_CREATOR_CHANNEL_PLAYLISTS:139,LATENCY_ACTION_CREATOR_CHANNEL_MUSIC_STOREFRONT:177,LATENCY_ACTION_CREATOR_CHANNEL_MUSIC:99,LATENCY_ACTION_CREATOR_CHANNEL_MONETIZATION:43,LATENCY_ACTION_CREATOR_CHANNEL_EDITING:113,LATENCY_ACTION_CREATOR_CHANNEL_DASHBOARD:49,
LATENCY_ACTION_CREATOR_CHANNEL_COPYRIGHT:44,LATENCY_ACTION_CREATOR_CMS_ISSUES:191,LATENCY_ACTION_CREATOR_CHANNEL_COMMENTS:66,LATENCY_ACTION_CREATOR_CHANNEL_ANALYTICS:31,LATENCY_ACTION_CREATOR_ARTIST_PROFILE:85,LATENCY_ACTION_CREATOR_ARTIST_CONCERTS:84,LATENCY_ACTION_CREATOR_ARTIST_ANALYTICS:83,LATENCY_ACTION_CREATOR_ANALYTICS_EXPLORE:140,LATENCY_ACTION_EXPERIMENTAL_WATCH_UI:181,LATENCY_ACTION_STORYBOARD_THUMBNAILS:118,LATENCY_ACTION_SEARCH_THUMBNAILS:59,LATENCY_ACTION_ON_DEVICE_MODEL_DOWNLOAD:54,
LATENCY_ACTION_VOICE_ASSISTANT:47,LATENCY_ACTION_SEARCH_UI:35,LATENCY_ACTION_SUGGEST:30,LATENCY_ACTION_AUTO_SEARCH:126,LATENCY_ACTION_DOWNLOADS:98,LATENCY_ACTION_EXPLORE:75,LATENCY_ACTION_VIDEO_LIST:63,LATENCY_ACTION_HOME_RESUME:60,LATENCY_ACTION_SUBSCRIPTIONS_LIST:57,LATENCY_ACTION_THUMBNAIL_LOAD:42,LATENCY_ACTION_FIRST_THUMBNAIL_LOAD:29,LATENCY_ACTION_SUBSCRIPTIONS_FEED:109,LATENCY_ACTION_SUBSCRIPTIONS:28,LATENCY_ACTION_TRENDING:27,LATENCY_ACTION_LIBRARY:21,LATENCY_ACTION_VIDEO_THUMBNAIL:8,LATENCY_ACTION_SHOW_MORE:7,
LATENCY_ACTION_VIDEO_PREVIEW:6,LATENCY_ACTION_AD_TO_AD:22,LATENCY_ACTION_VIDEO_TO_AD:17,LATENCY_ACTION_AD_TO_VIDEO:16,LATENCY_ACTION_DIRECT_PLAYBACK:132,LATENCY_ACTION_PREBUFFER_VIDEO:144,LATENCY_ACTION_PREFETCH_VIDEO:143,LATENCY_ACTION_STARTUP:106,LATENCY_ACTION_ONBOARDING:135,LATENCY_ACTION_LOGIN:97,LATENCY_ACTION_REEL_WATCH:41,LATENCY_ACTION_WATCH:3,LATENCY_ACTION_RESULTS:2,LATENCY_ACTION_CHANNELS:4,LATENCY_ACTION_HOME:1,LATENCY_ACTION_BROWSE:11,LATENCY_ACTION_USER_ACTION:189,LATENCY_ACTION_INFRASTRUCTURE:188,
LATENCY_ACTION_PAGE_NAVIGATION:187,LATENCY_ACTION_UNKNOWN:0};W[W.LATENCY_ACTION_SHORTS_VIDEO_INGESTION_TRANSCODING]="LATENCY_ACTION_SHORTS_VIDEO_INGESTION_TRANSCODING";W[W.LATENCY_ACTION_KIDS_PROFILE_SWITCHER]="LATENCY_ACTION_KIDS_PROFILE_SWITCHER";W[W.LATENCY_ACTION_OFFLINE_THUMBNAIL_TRANSFER]="LATENCY_ACTION_OFFLINE_THUMBNAIL_TRANSFER";W[W.LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR_ASYNC]="LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR_ASYNC";W[W.LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR]="LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR";
W[W.LATENCY_ACTION_SPINNER_DISPLAYED]="LATENCY_ACTION_SPINNER_DISPLAYED";W[W.LATENCY_ACTION_PLAYABILITY_CHECK]="LATENCY_ACTION_PLAYABILITY_CHECK";W[W.LATENCY_ACTION_PROCESS]="LATENCY_ACTION_PROCESS";W[W.LATENCY_ACTION_APP_STARTUP]="LATENCY_ACTION_APP_STARTUP";W[W.LATENCY_ACTION_COMMERCE_TRANSACTION]="LATENCY_ACTION_COMMERCE_TRANSACTION";W[W.LATENCY_ACTION_LOG_PAYMENT_SERVER_ANALYTICS_RPC]="LATENCY_ACTION_LOG_PAYMENT_SERVER_ANALYTICS_RPC";W[W.LATENCY_ACTION_GET_PAYMENT_INSTRUMENTS_PARAMS_RPC]="LATENCY_ACTION_GET_PAYMENT_INSTRUMENTS_PARAMS_RPC";
W[W.LATENCY_ACTION_GET_FIX_INSTRUMENT_PARAMS_RPC]="LATENCY_ACTION_GET_FIX_INSTRUMENT_PARAMS_RPC";W[W.LATENCY_ACTION_RESUME_SUBSCRIPTION_RPC]="LATENCY_ACTION_RESUME_SUBSCRIPTION_RPC";W[W.LATENCY_ACTION_PAUSE_SUBSCRIPTION_RPC]="LATENCY_ACTION_PAUSE_SUBSCRIPTION_RPC";W[W.LATENCY_ACTION_GET_OFFLINE_UPSELL_RPC]="LATENCY_ACTION_GET_OFFLINE_UPSELL_RPC";W[W.LATENCY_ACTION_GET_OFFERS_RPC]="LATENCY_ACTION_GET_OFFERS_RPC";W[W.LATENCY_ACTION_GET_CANCELLATION_YT_FLOW_RPC]="LATENCY_ACTION_GET_CANCELLATION_YT_FLOW_RPC";
W[W.LATENCY_ACTION_GET_CANCELLATION_FLOW_RPC]="LATENCY_ACTION_GET_CANCELLATION_FLOW_RPC";W[W.LATENCY_ACTION_UPDATE_CROSS_DEVICE_OFFLINE_STATE_RPC]="LATENCY_ACTION_UPDATE_CROSS_DEVICE_OFFLINE_STATE_RPC";W[W.LATENCY_ACTION_GET_OFFER_DETAILS_RPC]="LATENCY_ACTION_GET_OFFER_DETAILS_RPC";W[W.LATENCY_ACTION_CANCEL_RECURRENCE_TRANSACTION_RPC]="LATENCY_ACTION_CANCEL_RECURRENCE_TRANSACTION_RPC";W[W.LATENCY_ACTION_GET_TIP_MODULE_RPC]="LATENCY_ACTION_GET_TIP_MODULE_RPC";
W[W.LATENCY_ACTION_HANDLE_TRANSACTION_RPC]="LATENCY_ACTION_HANDLE_TRANSACTION_RPC";W[W.LATENCY_ACTION_COMPLETE_TRANSACTION_RPC]="LATENCY_ACTION_COMPLETE_TRANSACTION_RPC";W[W.LATENCY_ACTION_GET_CART_RPC]="LATENCY_ACTION_GET_CART_RPC";W[W.LATENCY_ACTION_THUMBNAIL_FETCH]="LATENCY_ACTION_THUMBNAIL_FETCH";W[W.LATENCY_ACTION_ABANDONED_DIRECT_PLAYBACK]="LATENCY_ACTION_ABANDONED_DIRECT_PLAYBACK";W[W.LATENCY_ACTION_SHARE_VIDEO]="LATENCY_ACTION_SHARE_VIDEO";W[W.LATENCY_ACTION_AD_TO_VIDEO_INT]="LATENCY_ACTION_AD_TO_VIDEO_INT";
W[W.LATENCY_ACTION_ABANDONED_BROWSE]="LATENCY_ACTION_ABANDONED_BROWSE";W[W.LATENCY_ACTION_PLAYER_ROTATION]="LATENCY_ACTION_PLAYER_ROTATION";W[W.LATENCY_ACTION_GENERIC_WEB_VIEW]="LATENCY_ACTION_GENERIC_WEB_VIEW";W[W.LATENCY_ACTION_SHOPPING_IN_APP]="LATENCY_ACTION_SHOPPING_IN_APP";W[W.LATENCY_ACTION_PLAYER_ATTESTATION]="LATENCY_ACTION_PLAYER_ATTESTATION";W[W.LATENCY_ACTION_PLAYER_SEEK]="LATENCY_ACTION_PLAYER_SEEK";W[W.LATENCY_ACTION_SUPER_STICKER_BUY_FLOW]="LATENCY_ACTION_SUPER_STICKER_BUY_FLOW";
W[W.LATENCY_ACTION_DOWNLOADS_DATA_ACCESS]="LATENCY_ACTION_DOWNLOADS_DATA_ACCESS";W[W.LATENCY_ACTION_BLOCKS_PERFORMANCE]="LATENCY_ACTION_BLOCKS_PERFORMANCE";W[W.LATENCY_ACTION_ASSISTANT_QUERY]="LATENCY_ACTION_ASSISTANT_QUERY";W[W.LATENCY_ACTION_ASSISTANT_SETTINGS]="LATENCY_ACTION_ASSISTANT_SETTINGS";W[W.LATENCY_ACTION_ENTITY_KEY_DESERIALIZATION_PERF]="LATENCY_ACTION_ENTITY_KEY_DESERIALIZATION_PERF";W[W.LATENCY_ACTION_ENTITY_KEY_SERIALIZATION_PERF]="LATENCY_ACTION_ENTITY_KEY_SERIALIZATION_PERF";
W[W.LATENCY_ACTION_PROOF_OF_ORIGIN_TOKEN_CREATE]="LATENCY_ACTION_PROOF_OF_ORIGIN_TOKEN_CREATE";W[W.LATENCY_ACTION_EMBEDS_SDK_INITIALIZATION]="LATENCY_ACTION_EMBEDS_SDK_INITIALIZATION";W[W.LATENCY_ACTION_NETWORKLESS_PERFORMANCE]="LATENCY_ACTION_NETWORKLESS_PERFORMANCE";W[W.LATENCY_ACTION_DOWNLOADS_EXPANSION]="LATENCY_ACTION_DOWNLOADS_EXPANSION";W[W.LATENCY_ACTION_ENTITY_TRANSFORM]="LATENCY_ACTION_ENTITY_TRANSFORM";W[W.LATENCY_ACTION_DOWNLOADS_COMPATIBILITY_LAYER]="LATENCY_ACTION_DOWNLOADS_COMPATIBILITY_LAYER";
W[W.LATENCY_ACTION_EMBEDS_SET_VIDEO]="LATENCY_ACTION_EMBEDS_SET_VIDEO";W[W.LATENCY_ACTION_SETTINGS]="LATENCY_ACTION_SETTINGS";W[W.LATENCY_ACTION_ABANDONED_STARTUP]="LATENCY_ACTION_ABANDONED_STARTUP";W[W.LATENCY_ACTION_MEDIA_BROWSER_ALARM_PLAY]="LATENCY_ACTION_MEDIA_BROWSER_ALARM_PLAY";W[W.LATENCY_ACTION_MEDIA_BROWSER_SEARCH]="LATENCY_ACTION_MEDIA_BROWSER_SEARCH";W[W.LATENCY_ACTION_MEDIA_BROWSER_LOAD_TREE]="LATENCY_ACTION_MEDIA_BROWSER_LOAD_TREE";W[W.LATENCY_ACTION_WHO_IS_WATCHING]="LATENCY_ACTION_WHO_IS_WATCHING";
W[W.LATENCY_ACTION_CAST_LOAD_BY_ENTITY_TO_WATCH]="LATENCY_ACTION_CAST_LOAD_BY_ENTITY_TO_WATCH";W[W.LATENCY_ACTION_LITE_SWITCH_ACCOUNT]="LATENCY_ACTION_LITE_SWITCH_ACCOUNT";W[W.LATENCY_ACTION_ELEMENTS_PERFORMANCE]="LATENCY_ACTION_ELEMENTS_PERFORMANCE";W[W.LATENCY_ACTION_LOCATION_SIGNAL_COLLECTION]="LATENCY_ACTION_LOCATION_SIGNAL_COLLECTION";W[W.LATENCY_ACTION_MODIFY_CHANNEL_NOTIFICATION]="LATENCY_ACTION_MODIFY_CHANNEL_NOTIFICATION";W[W.LATENCY_ACTION_OFFLINE_STORE_START]="LATENCY_ACTION_OFFLINE_STORE_START";
W[W.LATENCY_ACTION_REEL_EDITOR]="LATENCY_ACTION_REEL_EDITOR";W[W.LATENCY_ACTION_CHANNEL_SUBSCRIBE]="LATENCY_ACTION_CHANNEL_SUBSCRIBE";W[W.LATENCY_ACTION_CHANNEL_PREVIEW]="LATENCY_ACTION_CHANNEL_PREVIEW";W[W.LATENCY_ACTION_PREFETCH]="LATENCY_ACTION_PREFETCH";W[W.LATENCY_ACTION_ABANDONED_WATCH]="LATENCY_ACTION_ABANDONED_WATCH";W[W.LATENCY_ACTION_LOAD_COMMENT_REPLIES]="LATENCY_ACTION_LOAD_COMMENT_REPLIES";W[W.LATENCY_ACTION_LOAD_COMMENTS]="LATENCY_ACTION_LOAD_COMMENTS";
W[W.LATENCY_ACTION_EDIT_COMMENT]="LATENCY_ACTION_EDIT_COMMENT";W[W.LATENCY_ACTION_NEW_COMMENT]="LATENCY_ACTION_NEW_COMMENT";W[W.LATENCY_ACTION_OFFLINE_SHARING_RECEIVER_PAIRING]="LATENCY_ACTION_OFFLINE_SHARING_RECEIVER_PAIRING";W[W.LATENCY_ACTION_EMBED]="LATENCY_ACTION_EMBED";W[W.LATENCY_ACTION_MDX_LAUNCH]="LATENCY_ACTION_MDX_LAUNCH";W[W.LATENCY_ACTION_RESOLVE_URL]="LATENCY_ACTION_RESOLVE_URL";W[W.LATENCY_ACTION_CAST_SPLASH]="LATENCY_ACTION_CAST_SPLASH";W[W.LATENCY_ACTION_MDX_CONNECT_TO_SESSION]="LATENCY_ACTION_MDX_CONNECT_TO_SESSION";
W[W.LATENCY_ACTION_MDX_STREAM_TRANSFER]="LATENCY_ACTION_MDX_STREAM_TRANSFER";W[W.LATENCY_ACTION_MDX_CAST]="LATENCY_ACTION_MDX_CAST";W[W.LATENCY_ACTION_MDX_COMMAND]="LATENCY_ACTION_MDX_COMMAND";W[W.LATENCY_ACTION_REEL_SELECT_SEGMENT]="LATENCY_ACTION_REEL_SELECT_SEGMENT";W[W.LATENCY_ACTION_ACCELERATED_EFFECTS]="LATENCY_ACTION_ACCELERATED_EFFECTS";W[W.LATENCY_ACTION_EDIT_AUDIO_GEN]="LATENCY_ACTION_EDIT_AUDIO_GEN";W[W.LATENCY_ACTION_UPLOAD_AUDIO_MIXER]="LATENCY_ACTION_UPLOAD_AUDIO_MIXER";
W[W.LATENCY_ACTION_SHORTS_CLIENT_SIDE_RENDERING]="LATENCY_ACTION_SHORTS_CLIENT_SIDE_RENDERING";W[W.LATENCY_ACTION_SHORTS_SEG_IMP_TRANSCODING]="LATENCY_ACTION_SHORTS_SEG_IMP_TRANSCODING";W[W.LATENCY_ACTION_SHORTS_AUDIO_PICKER_PLAYBACK]="LATENCY_ACTION_SHORTS_AUDIO_PICKER_PLAYBACK";W[W.LATENCY_ACTION_SHORTS_WAVEFORM_DOWNLOAD]="LATENCY_ACTION_SHORTS_WAVEFORM_DOWNLOAD";W[W.LATENCY_ACTION_SHORTS_VIDEO_INGESTION]="LATENCY_ACTION_SHORTS_VIDEO_INGESTION";W[W.LATENCY_ACTION_SHORTS_GALLERY]="LATENCY_ACTION_SHORTS_GALLERY";
W[W.LATENCY_ACTION_SHORTS_TRIM]="LATENCY_ACTION_SHORTS_TRIM";W[W.LATENCY_ACTION_SHORTS_EDIT]="LATENCY_ACTION_SHORTS_EDIT";W[W.LATENCY_ACTION_SHORTS_CAMERA]="LATENCY_ACTION_SHORTS_CAMERA";W[W.LATENCY_ACTION_PRODUCER_EXPORT_PROJECT]="LATENCY_ACTION_PRODUCER_EXPORT_PROJECT";W[W.LATENCY_ACTION_PRODUCER_EDITOR]="LATENCY_ACTION_PRODUCER_EDITOR";W[W.LATENCY_ACTION_PARENT_TOOLS_DASHBOARD]="LATENCY_ACTION_PARENT_TOOLS_DASHBOARD";W[W.LATENCY_ACTION_PARENT_TOOLS_COLLECTION]="LATENCY_ACTION_PARENT_TOOLS_COLLECTION";
W[W.LATENCY_ACTION_MUSIC_LOAD_RECOMMENDED_MEDIA_ITEMS]="LATENCY_ACTION_MUSIC_LOAD_RECOMMENDED_MEDIA_ITEMS";W[W.LATENCY_ACTION_MUSIC_LOAD_MEDIA_ITEMS]="LATENCY_ACTION_MUSIC_LOAD_MEDIA_ITEMS";W[W.LATENCY_ACTION_MUSIC_ALBUM_DETAIL]="LATENCY_ACTION_MUSIC_ALBUM_DETAIL";W[W.LATENCY_ACTION_MUSIC_PLAYLIST_DETAIL]="LATENCY_ACTION_MUSIC_PLAYLIST_DETAIL";W[W.LATENCY_ACTION_STORE]="LATENCY_ACTION_STORE";W[W.LATENCY_ACTION_CHIPS]="LATENCY_ACTION_CHIPS";W[W.LATENCY_ACTION_SEARCH_ZERO_STATE]="LATENCY_ACTION_SEARCH_ZERO_STATE";
W[W.LATENCY_ACTION_LIVE_PAGINATION]="LATENCY_ACTION_LIVE_PAGINATION";W[W.LATENCY_ACTION_LIVE]="LATENCY_ACTION_LIVE";W[W.LATENCY_ACTION_PREBUFFER]="LATENCY_ACTION_PREBUFFER";W[W.LATENCY_ACTION_TENX]="LATENCY_ACTION_TENX";W[W.LATENCY_ACTION_KIDS_PROFILE_SETTINGS]="LATENCY_ACTION_KIDS_PROFILE_SETTINGS";W[W.LATENCY_ACTION_KIDS_WATCH_IT_AGAIN]="LATENCY_ACTION_KIDS_WATCH_IT_AGAIN";W[W.LATENCY_ACTION_KIDS_SECRET_CODE]="LATENCY_ACTION_KIDS_SECRET_CODE";W[W.LATENCY_ACTION_KIDS_PARENT_PROFILE_SETTINGS]="LATENCY_ACTION_KIDS_PARENT_PROFILE_SETTINGS";
W[W.LATENCY_ACTION_KIDS_ONBOARDING]="LATENCY_ACTION_KIDS_ONBOARDING";W[W.LATENCY_ACTION_KIDS_VOICE_SEARCH]="LATENCY_ACTION_KIDS_VOICE_SEARCH";W[W.LATENCY_ACTION_KIDS_CURATED_COLLECTION]="LATENCY_ACTION_KIDS_CURATED_COLLECTION";W[W.LATENCY_ACTION_KIDS_LIBRARY]="LATENCY_ACTION_KIDS_LIBRARY";W[W.LATENCY_ACTION_CREATOR_PROMOTION_LIST]="LATENCY_ACTION_CREATOR_PROMOTION_LIST";W[W.LATENCY_ACTION_CREATOR_PROMOTION_EDIT]="LATENCY_ACTION_CREATOR_PROMOTION_EDIT";
W[W.LATENCY_ACTION_CREATOR_VIDEO_TRANSLATIONS]="LATENCY_ACTION_CREATOR_VIDEO_TRANSLATIONS";W[W.LATENCY_ACTION_CREATOR_VIDEO_MONETIZATION]="LATENCY_ACTION_CREATOR_VIDEO_MONETIZATION";W[W.LATENCY_ACTION_CREATOR_VIDEO_LIVE_STREAMING]="LATENCY_ACTION_CREATOR_VIDEO_LIVE_STREAMING";W[W.LATENCY_ACTION_CREATOR_VIDEO_LIVE_SETTINGS]="LATENCY_ACTION_CREATOR_VIDEO_LIVE_SETTINGS";W[W.LATENCY_ACTION_CREATOR_VIDEO_EDITOR_ASYNC]="LATENCY_ACTION_CREATOR_VIDEO_EDITOR_ASYNC";
W[W.LATENCY_ACTION_CREATOR_VIDEO_EDITOR]="LATENCY_ACTION_CREATOR_VIDEO_EDITOR";W[W.LATENCY_ACTION_CREATOR_VIDEO_EDIT]="LATENCY_ACTION_CREATOR_VIDEO_EDIT";W[W.LATENCY_ACTION_CREATOR_VIDEO_COMMENTS]="LATENCY_ACTION_CREATOR_VIDEO_COMMENTS";W[W.LATENCY_ACTION_CREATOR_VIDEO_ANALYTICS]="LATENCY_ACTION_CREATOR_VIDEO_ANALYTICS";W[W.LATENCY_ACTION_CREATOR_SONG_ANALYTICS]="LATENCY_ACTION_CREATOR_SONG_ANALYTICS";W[W.LATENCY_ACTION_CREATOR_POST_LIST]="LATENCY_ACTION_CREATOR_POST_LIST";
W[W.LATENCY_ACTION_CREATOR_POST_EDIT]="LATENCY_ACTION_CREATOR_POST_EDIT";W[W.LATENCY_ACTION_CREATOR_POST_COMMENTS]="LATENCY_ACTION_CREATOR_POST_COMMENTS";W[W.LATENCY_ACTION_CREATOR_LIVE_STREAMING]="LATENCY_ACTION_CREATOR_LIVE_STREAMING";W[W.LATENCY_ACTION_CREATOR_DIALOG_VIDEO_COPYRIGHT]="LATENCY_ACTION_CREATOR_DIALOG_VIDEO_COPYRIGHT";W[W.LATENCY_ACTION_CREATOR_DIALOG_UPLOADS]="LATENCY_ACTION_CREATOR_DIALOG_UPLOADS";W[W.LATENCY_ACTION_CREATOR_DIALOG_COPYRIGHT_STRIKES]="LATENCY_ACTION_CREATOR_DIALOG_COPYRIGHT_STRIKES";
W[W.LATENCY_ACTION_CREATOR_CHANNEL_VIDEOS]="LATENCY_ACTION_CREATOR_CHANNEL_VIDEOS";W[W.LATENCY_ACTION_CREATOR_CHANNEL_TRANSLATIONS]="LATENCY_ACTION_CREATOR_CHANNEL_TRANSLATIONS";W[W.LATENCY_ACTION_CREATOR_CHANNEL_PLAYLISTS]="LATENCY_ACTION_CREATOR_CHANNEL_PLAYLISTS";W[W.LATENCY_ACTION_CREATOR_CHANNEL_MUSIC_STOREFRONT]="LATENCY_ACTION_CREATOR_CHANNEL_MUSIC_STOREFRONT";W[W.LATENCY_ACTION_CREATOR_CHANNEL_MUSIC]="LATENCY_ACTION_CREATOR_CHANNEL_MUSIC";W[W.LATENCY_ACTION_CREATOR_CHANNEL_MONETIZATION]="LATENCY_ACTION_CREATOR_CHANNEL_MONETIZATION";
W[W.LATENCY_ACTION_CREATOR_CHANNEL_EDITING]="LATENCY_ACTION_CREATOR_CHANNEL_EDITING";W[W.LATENCY_ACTION_CREATOR_CHANNEL_DASHBOARD]="LATENCY_ACTION_CREATOR_CHANNEL_DASHBOARD";W[W.LATENCY_ACTION_CREATOR_CHANNEL_COPYRIGHT]="LATENCY_ACTION_CREATOR_CHANNEL_COPYRIGHT";W[W.LATENCY_ACTION_CREATOR_CMS_ISSUES]="LATENCY_ACTION_CREATOR_CMS_ISSUES";W[W.LATENCY_ACTION_CREATOR_CHANNEL_COMMENTS]="LATENCY_ACTION_CREATOR_CHANNEL_COMMENTS";W[W.LATENCY_ACTION_CREATOR_CHANNEL_ANALYTICS]="LATENCY_ACTION_CREATOR_CHANNEL_ANALYTICS";
W[W.LATENCY_ACTION_CREATOR_ARTIST_PROFILE]="LATENCY_ACTION_CREATOR_ARTIST_PROFILE";W[W.LATENCY_ACTION_CREATOR_ARTIST_CONCERTS]="LATENCY_ACTION_CREATOR_ARTIST_CONCERTS";W[W.LATENCY_ACTION_CREATOR_ARTIST_ANALYTICS]="LATENCY_ACTION_CREATOR_ARTIST_ANALYTICS";W[W.LATENCY_ACTION_CREATOR_ANALYTICS_EXPLORE]="LATENCY_ACTION_CREATOR_ANALYTICS_EXPLORE";W[W.LATENCY_ACTION_EXPERIMENTAL_WATCH_UI]="LATENCY_ACTION_EXPERIMENTAL_WATCH_UI";W[W.LATENCY_ACTION_STORYBOARD_THUMBNAILS]="LATENCY_ACTION_STORYBOARD_THUMBNAILS";
W[W.LATENCY_ACTION_SEARCH_THUMBNAILS]="LATENCY_ACTION_SEARCH_THUMBNAILS";W[W.LATENCY_ACTION_ON_DEVICE_MODEL_DOWNLOAD]="LATENCY_ACTION_ON_DEVICE_MODEL_DOWNLOAD";W[W.LATENCY_ACTION_VOICE_ASSISTANT]="LATENCY_ACTION_VOICE_ASSISTANT";W[W.LATENCY_ACTION_SEARCH_UI]="LATENCY_ACTION_SEARCH_UI";W[W.LATENCY_ACTION_SUGGEST]="LATENCY_ACTION_SUGGEST";W[W.LATENCY_ACTION_AUTO_SEARCH]="LATENCY_ACTION_AUTO_SEARCH";W[W.LATENCY_ACTION_DOWNLOADS]="LATENCY_ACTION_DOWNLOADS";W[W.LATENCY_ACTION_EXPLORE]="LATENCY_ACTION_EXPLORE";
W[W.LATENCY_ACTION_VIDEO_LIST]="LATENCY_ACTION_VIDEO_LIST";W[W.LATENCY_ACTION_HOME_RESUME]="LATENCY_ACTION_HOME_RESUME";W[W.LATENCY_ACTION_SUBSCRIPTIONS_LIST]="LATENCY_ACTION_SUBSCRIPTIONS_LIST";W[W.LATENCY_ACTION_THUMBNAIL_LOAD]="LATENCY_ACTION_THUMBNAIL_LOAD";W[W.LATENCY_ACTION_FIRST_THUMBNAIL_LOAD]="LATENCY_ACTION_FIRST_THUMBNAIL_LOAD";W[W.LATENCY_ACTION_SUBSCRIPTIONS_FEED]="LATENCY_ACTION_SUBSCRIPTIONS_FEED";W[W.LATENCY_ACTION_SUBSCRIPTIONS]="LATENCY_ACTION_SUBSCRIPTIONS";
W[W.LATENCY_ACTION_TRENDING]="LATENCY_ACTION_TRENDING";W[W.LATENCY_ACTION_LIBRARY]="LATENCY_ACTION_LIBRARY";W[W.LATENCY_ACTION_VIDEO_THUMBNAIL]="LATENCY_ACTION_VIDEO_THUMBNAIL";W[W.LATENCY_ACTION_SHOW_MORE]="LATENCY_ACTION_SHOW_MORE";W[W.LATENCY_ACTION_VIDEO_PREVIEW]="LATENCY_ACTION_VIDEO_PREVIEW";W[W.LATENCY_ACTION_AD_TO_AD]="LATENCY_ACTION_AD_TO_AD";W[W.LATENCY_ACTION_VIDEO_TO_AD]="LATENCY_ACTION_VIDEO_TO_AD";W[W.LATENCY_ACTION_AD_TO_VIDEO]="LATENCY_ACTION_AD_TO_VIDEO";
W[W.LATENCY_ACTION_DIRECT_PLAYBACK]="LATENCY_ACTION_DIRECT_PLAYBACK";W[W.LATENCY_ACTION_PREBUFFER_VIDEO]="LATENCY_ACTION_PREBUFFER_VIDEO";W[W.LATENCY_ACTION_PREFETCH_VIDEO]="LATENCY_ACTION_PREFETCH_VIDEO";W[W.LATENCY_ACTION_STARTUP]="LATENCY_ACTION_STARTUP";W[W.LATENCY_ACTION_ONBOARDING]="LATENCY_ACTION_ONBOARDING";W[W.LATENCY_ACTION_LOGIN]="LATENCY_ACTION_LOGIN";W[W.LATENCY_ACTION_REEL_WATCH]="LATENCY_ACTION_REEL_WATCH";W[W.LATENCY_ACTION_WATCH]="LATENCY_ACTION_WATCH";
W[W.LATENCY_ACTION_RESULTS]="LATENCY_ACTION_RESULTS";W[W.LATENCY_ACTION_CHANNELS]="LATENCY_ACTION_CHANNELS";W[W.LATENCY_ACTION_HOME]="LATENCY_ACTION_HOME";W[W.LATENCY_ACTION_BROWSE]="LATENCY_ACTION_BROWSE";W[W.LATENCY_ACTION_USER_ACTION]="LATENCY_ACTION_USER_ACTION";W[W.LATENCY_ACTION_INFRASTRUCTURE]="LATENCY_ACTION_INFRASTRUCTURE";W[W.LATENCY_ACTION_PAGE_NAVIGATION]="LATENCY_ACTION_PAGE_NAVIGATION";W[W.LATENCY_ACTION_UNKNOWN]="LATENCY_ACTION_UNKNOWN";
var gu={LATENCY_NETWORK_MOBILE:2,LATENCY_NETWORK_WIFI:1,LATENCY_NETWORK_UNKNOWN:0};gu[gu.LATENCY_NETWORK_MOBILE]="LATENCY_NETWORK_MOBILE";gu[gu.LATENCY_NETWORK_WIFI]="LATENCY_NETWORK_WIFI";gu[gu.LATENCY_NETWORK_UNKNOWN]="LATENCY_NETWORK_UNKNOWN";var X={CONN_INVALID:31,CONN_CELLULAR_5G_NSA:12,CONN_CELLULAR_5G_SA:11,CONN_WIFI_METERED:10,CONN_CELLULAR_5G:9,CONN_DISCO:8,CONN_CELLULAR_UNKNOWN:7,CONN_CELLULAR_4G:6,CONN_CELLULAR_3G:5,CONN_CELLULAR_2G:4,CONN_WIFI:3,CONN_NONE:2,CONN_UNKNOWN:1,CONN_DEFAULT:0};
X[X.CONN_INVALID]="CONN_INVALID";X[X.CONN_CELLULAR_5G_NSA]="CONN_CELLULAR_5G_NSA";X[X.CONN_CELLULAR_5G_SA]="CONN_CELLULAR_5G_SA";X[X.CONN_WIFI_METERED]="CONN_WIFI_METERED";X[X.CONN_CELLULAR_5G]="CONN_CELLULAR_5G";X[X.CONN_DISCO]="CONN_DISCO";X[X.CONN_CELLULAR_UNKNOWN]="CONN_CELLULAR_UNKNOWN";X[X.CONN_CELLULAR_4G]="CONN_CELLULAR_4G";X[X.CONN_CELLULAR_3G]="CONN_CELLULAR_3G";X[X.CONN_CELLULAR_2G]="CONN_CELLULAR_2G";X[X.CONN_WIFI]="CONN_WIFI";X[X.CONN_NONE]="CONN_NONE";X[X.CONN_UNKNOWN]="CONN_UNKNOWN";
X[X.CONN_DEFAULT]="CONN_DEFAULT";
var Y={DETAILED_NETWORK_TYPE_NR_NSA:126,DETAILED_NETWORK_TYPE_NR_SA:125,DETAILED_NETWORK_TYPE_INTERNAL_WIFI_IMPAIRED:124,DETAILED_NETWORK_TYPE_APP_WIFI_HOTSPOT:123,DETAILED_NETWORK_TYPE_DISCONNECTED:122,DETAILED_NETWORK_TYPE_NON_MOBILE_UNKNOWN:121,DETAILED_NETWORK_TYPE_MOBILE_UNKNOWN:120,DETAILED_NETWORK_TYPE_WIMAX:119,DETAILED_NETWORK_TYPE_ETHERNET:118,DETAILED_NETWORK_TYPE_BLUETOOTH:117,DETAILED_NETWORK_TYPE_WIFI:116,DETAILED_NETWORK_TYPE_LTE:115,DETAILED_NETWORK_TYPE_HSPAP:114,DETAILED_NETWORK_TYPE_EHRPD:113,
DETAILED_NETWORK_TYPE_EVDO_B:112,DETAILED_NETWORK_TYPE_UMTS:111,DETAILED_NETWORK_TYPE_IDEN:110,DETAILED_NETWORK_TYPE_HSUPA:109,DETAILED_NETWORK_TYPE_HSPA:108,DETAILED_NETWORK_TYPE_HSDPA:107,DETAILED_NETWORK_TYPE_EVDO_A:106,DETAILED_NETWORK_TYPE_EVDO_0:105,DETAILED_NETWORK_TYPE_CDMA:104,DETAILED_NETWORK_TYPE_1_X_RTT:103,DETAILED_NETWORK_TYPE_GPRS:102,DETAILED_NETWORK_TYPE_EDGE:101,DETAILED_NETWORK_TYPE_UNKNOWN:0};Y[Y.DETAILED_NETWORK_TYPE_NR_NSA]="DETAILED_NETWORK_TYPE_NR_NSA";
Y[Y.DETAILED_NETWORK_TYPE_NR_SA]="DETAILED_NETWORK_TYPE_NR_SA";Y[Y.DETAILED_NETWORK_TYPE_INTERNAL_WIFI_IMPAIRED]="DETAILED_NETWORK_TYPE_INTERNAL_WIFI_IMPAIRED";Y[Y.DETAILED_NETWORK_TYPE_APP_WIFI_HOTSPOT]="DETAILED_NETWORK_TYPE_APP_WIFI_HOTSPOT";Y[Y.DETAILED_NETWORK_TYPE_DISCONNECTED]="DETAILED_NETWORK_TYPE_DISCONNECTED";Y[Y.DETAILED_NETWORK_TYPE_NON_MOBILE_UNKNOWN]="DETAILED_NETWORK_TYPE_NON_MOBILE_UNKNOWN";Y[Y.DETAILED_NETWORK_TYPE_MOBILE_UNKNOWN]="DETAILED_NETWORK_TYPE_MOBILE_UNKNOWN";
Y[Y.DETAILED_NETWORK_TYPE_WIMAX]="DETAILED_NETWORK_TYPE_WIMAX";Y[Y.DETAILED_NETWORK_TYPE_ETHERNET]="DETAILED_NETWORK_TYPE_ETHERNET";Y[Y.DETAILED_NETWORK_TYPE_BLUETOOTH]="DETAILED_NETWORK_TYPE_BLUETOOTH";Y[Y.DETAILED_NETWORK_TYPE_WIFI]="DETAILED_NETWORK_TYPE_WIFI";Y[Y.DETAILED_NETWORK_TYPE_LTE]="DETAILED_NETWORK_TYPE_LTE";Y[Y.DETAILED_NETWORK_TYPE_HSPAP]="DETAILED_NETWORK_TYPE_HSPAP";Y[Y.DETAILED_NETWORK_TYPE_EHRPD]="DETAILED_NETWORK_TYPE_EHRPD";Y[Y.DETAILED_NETWORK_TYPE_EVDO_B]="DETAILED_NETWORK_TYPE_EVDO_B";
Y[Y.DETAILED_NETWORK_TYPE_UMTS]="DETAILED_NETWORK_TYPE_UMTS";Y[Y.DETAILED_NETWORK_TYPE_IDEN]="DETAILED_NETWORK_TYPE_IDEN";Y[Y.DETAILED_NETWORK_TYPE_HSUPA]="DETAILED_NETWORK_TYPE_HSUPA";Y[Y.DETAILED_NETWORK_TYPE_HSPA]="DETAILED_NETWORK_TYPE_HSPA";Y[Y.DETAILED_NETWORK_TYPE_HSDPA]="DETAILED_NETWORK_TYPE_HSDPA";Y[Y.DETAILED_NETWORK_TYPE_EVDO_A]="DETAILED_NETWORK_TYPE_EVDO_A";Y[Y.DETAILED_NETWORK_TYPE_EVDO_0]="DETAILED_NETWORK_TYPE_EVDO_0";Y[Y.DETAILED_NETWORK_TYPE_CDMA]="DETAILED_NETWORK_TYPE_CDMA";
Y[Y.DETAILED_NETWORK_TYPE_1_X_RTT]="DETAILED_NETWORK_TYPE_1_X_RTT";Y[Y.DETAILED_NETWORK_TYPE_GPRS]="DETAILED_NETWORK_TYPE_GPRS";Y[Y.DETAILED_NETWORK_TYPE_EDGE]="DETAILED_NETWORK_TYPE_EDGE";Y[Y.DETAILED_NETWORK_TYPE_UNKNOWN]="DETAILED_NETWORK_TYPE_UNKNOWN";var hu={LATENCY_PLAYER_RTSP:7,LATENCY_PLAYER_HTML5_INLINE:6,LATENCY_PLAYER_HTML5_FULLSCREEN:5,LATENCY_PLAYER_HTML5:4,LATENCY_PLAYER_FRAMEWORK:3,LATENCY_PLAYER_FLASH:2,LATENCY_PLAYER_EXO:1,LATENCY_PLAYER_UNKNOWN:0};hu[hu.LATENCY_PLAYER_RTSP]="LATENCY_PLAYER_RTSP";
hu[hu.LATENCY_PLAYER_HTML5_INLINE]="LATENCY_PLAYER_HTML5_INLINE";hu[hu.LATENCY_PLAYER_HTML5_FULLSCREEN]="LATENCY_PLAYER_HTML5_FULLSCREEN";hu[hu.LATENCY_PLAYER_HTML5]="LATENCY_PLAYER_HTML5";hu[hu.LATENCY_PLAYER_FRAMEWORK]="LATENCY_PLAYER_FRAMEWORK";hu[hu.LATENCY_PLAYER_FLASH]="LATENCY_PLAYER_FLASH";hu[hu.LATENCY_PLAYER_EXO]="LATENCY_PLAYER_EXO";hu[hu.LATENCY_PLAYER_UNKNOWN]="LATENCY_PLAYER_UNKNOWN";
var iu={LATENCY_AD_BREAK_TYPE_POSTROLL:3,LATENCY_AD_BREAK_TYPE_MIDROLL:2,LATENCY_AD_BREAK_TYPE_PREROLL:1,LATENCY_AD_BREAK_TYPE_UNKNOWN:0};iu[iu.LATENCY_AD_BREAK_TYPE_POSTROLL]="LATENCY_AD_BREAK_TYPE_POSTROLL";iu[iu.LATENCY_AD_BREAK_TYPE_MIDROLL]="LATENCY_AD_BREAK_TYPE_MIDROLL";iu[iu.LATENCY_AD_BREAK_TYPE_PREROLL]="LATENCY_AD_BREAK_TYPE_PREROLL";iu[iu.LATENCY_AD_BREAK_TYPE_UNKNOWN]="LATENCY_AD_BREAK_TYPE_UNKNOWN";var ju={LATENCY_ACTION_ERROR_STARTUP_TIMEOUT:1,LATENCY_ACTION_ERROR_UNSPECIFIED:0};
ju[ju.LATENCY_ACTION_ERROR_STARTUP_TIMEOUT]="LATENCY_ACTION_ERROR_STARTUP_TIMEOUT";ju[ju.LATENCY_ACTION_ERROR_UNSPECIFIED]="LATENCY_ACTION_ERROR_UNSPECIFIED";var ku={LIVE_STREAM_MODE_WINDOW:5,LIVE_STREAM_MODE_POST:4,LIVE_STREAM_MODE_LP:3,LIVE_STREAM_MODE_LIVE:2,LIVE_STREAM_MODE_DVR:1,LIVE_STREAM_MODE_UNKNOWN:0};ku[ku.LIVE_STREAM_MODE_WINDOW]="LIVE_STREAM_MODE_WINDOW";ku[ku.LIVE_STREAM_MODE_POST]="LIVE_STREAM_MODE_POST";ku[ku.LIVE_STREAM_MODE_LP]="LIVE_STREAM_MODE_LP";
ku[ku.LIVE_STREAM_MODE_LIVE]="LIVE_STREAM_MODE_LIVE";ku[ku.LIVE_STREAM_MODE_DVR]="LIVE_STREAM_MODE_DVR";ku[ku.LIVE_STREAM_MODE_UNKNOWN]="LIVE_STREAM_MODE_UNKNOWN";var lu={VIDEO_STREAM_TYPE_VOD:3,VIDEO_STREAM_TYPE_DVR:2,VIDEO_STREAM_TYPE_LIVE:1,VIDEO_STREAM_TYPE_UNSPECIFIED:0};lu[lu.VIDEO_STREAM_TYPE_VOD]="VIDEO_STREAM_TYPE_VOD";lu[lu.VIDEO_STREAM_TYPE_DVR]="VIDEO_STREAM_TYPE_DVR";lu[lu.VIDEO_STREAM_TYPE_LIVE]="VIDEO_STREAM_TYPE_LIVE";lu[lu.VIDEO_STREAM_TYPE_UNSPECIFIED]="VIDEO_STREAM_TYPE_UNSPECIFIED";
var mu={YT_IDB_TRANSACTION_TYPE_READ:2,YT_IDB_TRANSACTION_TYPE_WRITE:1,YT_IDB_TRANSACTION_TYPE_UNKNOWN:0};mu[mu.YT_IDB_TRANSACTION_TYPE_READ]="YT_IDB_TRANSACTION_TYPE_READ";mu[mu.YT_IDB_TRANSACTION_TYPE_WRITE]="YT_IDB_TRANSACTION_TYPE_WRITE";mu[mu.YT_IDB_TRANSACTION_TYPE_UNKNOWN]="YT_IDB_TRANSACTION_TYPE_UNKNOWN";var nu={PLAYER_ROTATION_TYPE_PORTRAIT_TO_FULLSCREEN:2,PLAYER_ROTATION_TYPE_FULLSCREEN_TO_PORTRAIT:1,PLAYER_ROTATION_TYPE_UNKNOWN:0};nu[nu.PLAYER_ROTATION_TYPE_PORTRAIT_TO_FULLSCREEN]="PLAYER_ROTATION_TYPE_PORTRAIT_TO_FULLSCREEN";
nu[nu.PLAYER_ROTATION_TYPE_FULLSCREEN_TO_PORTRAIT]="PLAYER_ROTATION_TYPE_FULLSCREEN_TO_PORTRAIT";nu[nu.PLAYER_ROTATION_TYPE_UNKNOWN]="PLAYER_ROTATION_TYPE_UNKNOWN";var ou="actionVisualElement spinnerInfo resourceInfo playerInfo commentInfo mdxInfo watchInfo thumbnailLoadInfo creatorInfo unpluggedInfo reelInfo subscriptionsFeedInfo requestIds mediaBrowserActionInfo musicLoadActionInfo shoppingInfo webViewInfo prefetchInfo accelerationSession commerceInfo webInfo tvInfo kabukiInfo mwebInfo musicInfo".split(" ");var pu=z.ytLoggingLatencyUsageStats_||{};A("ytLoggingLatencyUsageStats_",pu);function qu(){this.h=0}
function ru(){qu.h||(qu.h=new qu);return qu.h}
qu.prototype.tick=function(a,b,c,d){su(this,"tick_"+a+"_"+b)||(c={timestamp:c,cttAuthInfo:d},M("web_csi_via_jspb")?(d=new Tj,D(d,1,a),D(d,2,b),a=new Wj,ae(a,Tj,5,Xj,d),Jr(a,c)):jm("latencyActionTicked",{tickName:a,clientActionNonce:b},c))};
qu.prototype.info=function(a,b,c){var d=Object.keys(a).join("");su(this,"info_"+d+"_"+b)||(a=Object.assign({},a),a.clientActionNonce=b,jm("latencyActionInfo",a,{cttAuthInfo:c}))};
qu.prototype.jspbInfo=function(a,b,c){for(var d="",e=0;e<a.toJSON().length;e++)void 0!==a.toJSON()[e]&&(d=0===e?d.concat(""+e):d.concat("_"+e));su(this,"info_"+d+"_"+b)||(D(a,2,b),b={cttAuthInfo:c},c=new Wj,ae(c,Pj,7,Xj,a),Jr(c,b))};
qu.prototype.span=function(a,b,c){var d=Object.keys(a).join("");su(this,"span_"+d+"_"+b)||(a.clientActionNonce=b,jm("latencyActionSpan",a,{cttAuthInfo:c}))};
function su(a,b){pu[b]=pu[b]||{count:0};var c=pu[b];c.count++;c.time=Q();a.h||(a.h=Rl(function(){var d=Q(),e;for(e in pu)pu[e]&&6E4<d-pu[e].time&&delete pu[e];a&&(a.h=0)},5E3));
return 5<c.count?(6===c.count&&1>1E5*Math.random()&&(c=new P("CSI data exceeded logging limit with key",b.split("_")),0<=b.indexOf("plev")||js(c)),!0):!1}
;function tu(){var a=["ol"];Yt("").info.actionType="embed";a&&Ik("TIMING_AFT_KEYS",a);Ik("TIMING_ACTION","embed");if(M("web_csi_via_jspb")){a=L("TIMING_INFO",{});var b=new Pj;a=p(Object.entries(a));for(var c=a.next();!c.done;c=a.next()){var d=p(c.value);c=d.next().value;d=d.next().value;switch(c){case "GetBrowse_rid":var e=new Sj;D(e,1,c);D(e,2,String(d));Rj(b,e);break;case "GetGuide_rid":e=new Sj;D(e,1,c);D(e,2,String(d));Rj(b,e);break;case "GetHome_rid":e=new Sj;D(e,1,c);D(e,2,String(d));Rj(b,e);
break;case "GetPlayer_rid":e=new Sj;D(e,1,c);D(e,2,String(d));Rj(b,e);break;case "GetSearch_rid":e=new Sj;D(e,1,c);D(e,2,String(d));Rj(b,e);break;case "GetSettings_rid":e=new Sj;D(e,1,c);D(e,2,String(d));Rj(b,e);break;case "GetTrending_rid":e=new Sj;D(e,1,c);D(e,2,String(d));Rj(b,e);break;case "GetWatchNext_rid":e=new Sj;D(e,1,c);D(e,2,String(d));Rj(b,e);break;case "yt_red":D(b,14,!!d);break;case "yt_ad":D(b,9,!!d)}}uu(b);b=new Pj;b=D(b,25,!0);b=D(b,1,W[eu(L("TIMING_ACTION"))]);(a=L("PREVIOUS_ACTION"))&&
D(b,13,W[eu(a)]);(a=L("CLIENT_PROTOCOL"))&&D(b,33,a);(a=L("CLIENT_TRANSPORT"))&&D(b,34,a);(a=Bs())&&"UNDEFINED_CSN"!==a&&D(b,4,a);a=vu();1!==a&&-1!==a||D(b,6,!0);a=Ot();D(b,3,"cold");wu(a);a=xu();if(0<a.length)for(a=p(a),c=a.next();!c.done;c=a.next())c=c.value,d=new Oj,D(d,1,c),ce(b,83,Oj,d);uu(b)}else{a=L("TIMING_INFO",{});for(b in a)a.hasOwnProperty(b)&&yu(b,a[b]);b={isNavigation:!0,actionType:eu(L("TIMING_ACTION"))};if(a=L("PREVIOUS_ACTION"))b.previousAction=eu(a);if(a=L("CLIENT_PROTOCOL"))b.httpProtocol=
a;if(a=L("CLIENT_TRANSPORT"))b.transportProtocol=a;(a=Bs())&&"UNDEFINED_CSN"!==a&&(b.clientScreenNonce=a);a=vu();if(1===a||-1===a)b.isVisible=!0;Ot();b.loadType="cold";wu(Ot());a=xu();if(0<a.length)for(b.resourceInfo=[],a=p(a),c=a.next();!c.done;c=a.next())b.resourceInfo.push({resourceCache:c.value});zu(b)}b=Ot();a=St();if(!(M("skip_setting_info_in_csi_data_object")&&"cold"!==Pt().loadType||"cold"!==b.yt_lt&&"cold"!==a.loadType)){c=Qt();d=Rt();d=d.gelTicks?d.gelTicks:d.gelTicks={};for(var f in c)if(!(f in
d))if("number"===typeof c[f])Z(f,Jt(f));else if(M("log_repeated_ytcsi_ticks")){e=p(c[f]);for(var g=e.next();!g.done;g=e.next())Z(f.slice(1),g.value)}f={};c=!1;d=p(Object.keys(b));for(e=d.next();!e.done;e=d.next())e=e.value,(e=fu(e,b[e]))&&!Vt(St(),e)&&(wt(a,e),wt(f,e),c=!0);c&&zu(f)}A("ytglobal.timingready_",!0);f=L("TIMING_ACTION");B("ytglobal.timingready_")&&f&&"_start"in Qt()&&It()&&Ut()}
function yu(a,b,c,d){if(null!==b){var e=Ot(c);M("skip_setting_info_in_csi_data_object")?"yt_lt"===a&&(e="string"===typeof b?b:""+b,Pt(c).loadType=e):e[a]=b;(a=fu(a,b,c))&&zu(a,c,d)}}
function zu(a,b,c){if(!M("web_csi_via_jspb")||(void 0===c?0:c))c=Yt(b||""),wt(c.info,a),M("skip_setting_info_in_csi_data_object")&&a.loadType&&(c=a.loadType,Pt(b).loadType=c),wt(St(b),a),c=Tt(b),b=Nt(b).cttAuthInfo,ru().info(a,c,b);else{c=new Pj;var d=Object.keys(a);a=Object.values(a);for(var e=0;e<d.length;e++){var f=d[e];try{switch(f){case "actionType":D(c,1,W[a[e]]);break;case "clientActionNonce":D(c,2,a[e]);break;case "clientScreenNonce":D(c,4,a[e]);break;case "loadType":D(c,3,a[e]);break;case "isPrewarmedLaunch":D(c,
92,a[e]);break;case "isFirstInstall":D(c,55,a[e]);break;case "networkType":D(c,5,gu[a[e]]);break;case "connectionType":D(c,26,X[a[e]]);break;case "detailedConnectionType":D(c,27,Y[a[e]]);break;case "isVisible":D(c,6,a[e]);break;case "playerType":D(c,7,hu[a[e]]);break;case "clientPlaybackNonce":D(c,8,a[e]);break;case "adClientPlaybackNonce":D(c,28,a[e]);break;case "previousCpn":D(c,77,a[e]);break;case "targetCpn":D(c,76,a[e]);break;case "isMonetized":D(c,9,a[e]);break;case "isPrerollAllowed":D(c,16,
a[e]);break;case "isPrerollShown":D(c,17,a[e]);break;case "adType":D(c,12,a[e]);break;case "adTypesAllowed":D(c,36,a[e]);break;case "adNetworks":D(c,37,a[e]);break;case "previousAction":D(c,13,W[a[e]]);break;case "isRedSubscriber":D(c,14,a[e]);break;case "serverTimeMs":D(c,15,a[e]);break;case "videoId":c.setVideoId(a[e]);break;case "adVideoId":D(c,20,a[e]);break;case "targetVideoId":D(c,78,a[e]);break;case "adBreakType":D(c,21,iu[a[e]]);break;case "isNavigation":D(c,25,a[e]);break;case "viewportHeight":D(c,
29,a[e]);break;case "viewportWidth":D(c,30,a[e]);break;case "screenHeight":D(c,84,a[e]);break;case "screenWidth":D(c,85,a[e]);break;case "browseId":D(c,31,a[e]);break;case "isCacheHit":D(c,32,a[e]);break;case "httpProtocol":D(c,33,a[e]);break;case "transportProtocol":D(c,34,a[e]);break;case "searchQuery":D(c,41,a[e]);break;case "isContinuation":D(c,42,a[e]);break;case "availableProcessors":D(c,43,a[e]);break;case "sdk":D(c,44,a[e]);break;case "isLocalStream":D(c,45,a[e]);break;case "navigationRequestedSameUrl":D(c,
64,a[e]);break;case "shellStartupDurationMs":D(c,70,a[e]);break;case "appInstallDataAgeMs":D(c,73,a[e]);break;case "latencyActionError":D(c,71,ju[a[e]]);break;case "actionStep":D(c,79,a[e]);break;case "jsHeapSizeLimit":D(c,80,a[e]);break;case "totalJsHeapSize":D(c,81,a[e]);break;case "usedJsHeapSize":D(c,82,a[e]);break;case "sourceVideoDurationMs":D(c,90,a[e]);break;case "videoOutputFrames":D(c,93,a[e]);break;case "isResume":D(c,104,a[e]);break;case "debugTicksExcluded":D(c,105,a[e]);break;case "adPrebufferedTimeSecs":D(c,
39,a[e]);break;case "isLivestream":D(c,47,a[e]);break;case "liveStreamMode":D(c,91,ku[a[e]]);break;case "adCpn2":D(c,48,a[e]);break;case "adDaiDriftMillis":D(c,49,a[e]);break;case "videoStreamType":D(c,53,lu[a[e]]);break;case "playbackRequiresTap":D(c,56,a[e]);break;case "performanceNavigationTiming":D(c,67,a[e]);break;case "transactionType":D(c,74,mu[a[e]]);break;case "playerRotationType":D(c,101,nu[a[e]]);break;case "allowedPreroll":D(c,10,a[e]);break;case "shownPreroll":D(c,11,a[e]);break;case "getHomeRequestId":D(c,
57,a[e]);break;case "getSearchRequestId":D(c,60,a[e]);break;case "getPlayerRequestId":D(c,61,a[e]);break;case "getWatchNextRequestId":D(c,62,a[e]);break;case "getBrowseRequestId":D(c,63,a[e]);break;case "getLibraryRequestId":D(c,66,a[e]);break;default:ou.includes(f)&&Rk(new P("Codegen laipb translator asked to translate message field",""+f))}}catch(g){Rk(Error("Codegen laipb translator failed to set "+f))}}uu(c,b)}}
function uu(a,b){if(M("skip_setting_info_in_csi_data_object")){var c=ee(Rd(a,3),"");c&&(Pt(b).loadType=c)}else c=Rt(b),c.jspbInfos||(c.jspbInfos=[]),c.jspbInfos.push(ke(a));Yt(b||"").jspbInfo.push(a);c=Tt(b);b=Nt(b).cttAuthInfo;ru().jspbInfo(a,c,b)}
function Z(a,b,c){if(!b&&"_"!==a[0]){var d=a;S.mark&&(0==d.lastIndexOf("mark_",0)||(d="mark_"+d),c&&(d+=" ("+c+")"),S.mark(d))}d=Yt(c||"");d.tick[a]=b||Q();if(d.callback&&d.callback[a]){d=p(d.callback[a]);for(var e=d.next();!e.done;e=d.next())e=e.value,e()}d=Rt(c);d.gelTicks&&(d.gelTicks[a]=!0);e=Qt(c);d=b||Q();M("log_repeated_ytcsi_ticks")?a in e||(e[a]=d):e[a]=d;e=Tt(c);var f=Nt(c).cttAuthInfo;"_start"===a?(a=ru(),su(a,"baseline_"+e)||(b={timestamp:b,cttAuthInfo:f},M("web_csi_via_jspb")?(a=new Nj,
D(a,1,e),e=new Wj,ae(e,Nj,6,Xj,a),Jr(e,b)):jm("latencyActionBaselined",{clientActionNonce:e},b))):ru().tick(a,e,b,f);Ut(c);return d}
function Au(){var a=Tt();requestAnimationFrame(function(){setTimeout(function(){a===Tt()&&Z("ol",void 0,void 0)},0)})}
function vu(){var a=document;if("visibilityState"in a)a=a.visibilityState;else{var b=up+"VisibilityState";a=b in a?a[b]:void 0}switch(a){case "hidden":return 0;case "visible":return 1;case "prerender":return 2;case "unloaded":return 3;default:return-1}}
function wu(a){var b=Kt(),c=Mt(),d=L("CSI_START_TIMESTAMP_MILLIS",0);0<d&&!M("embeds_web_enable_csi_start_override_killswitch")&&(c=d);c&&(Z("srt",b.responseStart),1!==a.prerender&&Z("_start",c,void 0));a=Wt();0<a&&Z("fpt",a);a=Kt();a.isPerformanceNavigationTiming&&zu({performanceNavigationTiming:!0});Z("nreqs",a.requestStart,void 0);Z("nress",a.responseStart,void 0);Z("nrese",a.responseEnd,void 0);0<a.redirectEnd-a.redirectStart&&(Z("nrs",a.redirectStart,void 0),Z("nre",a.redirectEnd,void 0));0<
a.domainLookupEnd-a.domainLookupStart&&(Z("ndnss",a.domainLookupStart,void 0),Z("ndnse",a.domainLookupEnd,void 0));0<a.connectEnd-a.connectStart&&(Z("ntcps",a.connectStart,void 0),Z("ntcpe",a.connectEnd,void 0));a.secureConnectionStart>=Mt()&&0<a.connectEnd-a.secureConnectionStart&&(Z("nstcps",a.secureConnectionStart,void 0),Z("ntcpe",a.connectEnd,void 0));S&&"getEntriesByType"in S&&Bu()}
function Cu(a,b){a=document.querySelector(a);if(!a)return!1;var c="",d=a.nodeName;"SCRIPT"===d?(c=a.src,c||(c=a.getAttribute("data-timing-href"))&&(c=window.location.protocol+c)):"LINK"===d&&(c=a.href);hc()&&a.setAttribute("nonce",hc());return c?(a=S.getEntriesByName(c))&&a[0]&&(a=a[0],c=Mt(),Z("rsf_"+b,c+Math.round(a.fetchStart)),Z("rse_"+b,c+Math.round(a.responseEnd)),void 0!==a.transferSize&&0===a.transferSize)?!0:!1:!1}
function xu(){var a=[];if(document.querySelector&&S&&S.getEntriesByName)for(var b in Ht)if(Ht.hasOwnProperty(b)){var c=Ht[b];Cu(b,c)&&a.push(c)}return a}
function Bu(){var a=window.location.protocol,b=S.getEntriesByType("resource");b=gb(b,function(c){return 0===c.name.indexOf(a+"//fonts.gstatic.com/s/")});
(b=ib(b,function(c,d){return d.duration>c.duration?d:c},{duration:0}))&&0<b.startTime&&0<b.responseEnd&&(Z("wffs",Lt(b.startTime)),Z("wffe",Lt(b.responseEnd)))}
var Du=window;Du.ytcsi&&(Du.ytcsi.info=yu,Du.ytcsi.tick=Z);var Eu="tokens consistency mss client_location entities response_received_commands store PLAYER_PRELOAD".split(" "),Fu=["type.googleapis.com/youtube.api.pfiinnertube.YoutubeApiInnertube.BrowseResponse"];function Gu(a,b,c,d){this.m=a;this.L=b;this.l=c;this.j=d;this.i=void 0;this.h=new Map;a.Qa||(a.Qa={});a.Qa=Object.assign({},vt,a.Qa)}
function Hu(a,b,c,d){if(void 0!==Gu.h){if(d=Gu.h,a=[a!==d.m,b!==d.L,c!==d.l,!1,!1,void 0!==d.i],a.some(function(e){return e}))throw new P("InnerTubeTransportService is already initialized",a);
}else Gu.h=new Gu(a,b,c,d)}
function Iu(a){var b={signalServiceEndpoint:{signal:"GET_DATASYNC_IDS"}};var c=void 0===c?dt:c;var d=nt(b,a.m);if(!d)return Ff(new P("Error: No request builder found for command.",b));var e=d.m(b,void 0,c);return e?new Af(function(f){var g,h,k;return x(function(m){if(1==m.h){h="cors"===(null==(g=e.xa)?void 0:g.mode)?"cors":void 0;if(a.l.wd){var q=e.config,r;q=null==q?void 0:null==(r=q.ab)?void 0:r.sessionIndex;r=ct({sessionIndex:q});k=Object.assign({},Ju(h),r);return m.u(2)}return v(m,Ku(e.config,
h),3)}2!=m.h&&(k=m.i);f(Lu(a,e,k));m.h=0})}):Ff(new P("Error: Failed to build request for command.",b))}
function Mu(a,b,c){var d;if(b&&!(null==b?0:null==(d=b.Gr)?0:d.Jr)&&a.j){d=p(Eu);for(var e=d.next();!e.done;e=d.next())e=e.value,a.j[e]&&a.j[e].handleResponse(b,c)}}
function Lu(a,b,c){var d,e,f,g,h,k,m,q,r,w,t,y,E,F,O,N,R,ca,U,rb,Va,na,I,ma,ea,Ee,Fe,sd;return x(function(pa){switch(pa.h){case 1:pa.u(2);break;case 3:if((d=pa.i)&&!d.isExpired())return pa.return(Promise.resolve(d.h()));case 2:if(null==(e=b)?0:null==(f=e.ga)?0:f.context)for(g=p([]),h=g.next();!h.done;h=g.next())k=h.value,k.Cr(b.ga.context);if(null==(m=a.i)||!m.Hr(b.input,b.ga)){pa.u(4);break}return v(pa,a.i.xr(b.input,b.ga),5);case 5:return q=pa.i,M("kevlar_process_local_innertube_responses_killswitch")||
Mu(a,q,b),pa.return(q);case 4:return(t=null==(w=b.config)?void 0:w.wa)&&a.h.has(t)&&M("web_memoize_inflight_requests")?r=a.h.get(t):(y=JSON.stringify(b.ga),O=null!=(F=null==(E=b.xa)?void 0:E.headers)?F:{},b.xa=Object.assign({},b.xa,{headers:Object.assign({},O,c)}),N=Object.assign({},b.xa),"POST"===b.xa.method&&(N=Object.assign({},N,{body:y})),(null==(R=b.config)?0:R.jd)&&Z(b.config.jd),ca=function(){return a.L.fetch(b.input,N,b.config)},r=ca(),t&&a.h.set(t,r)),v(pa,r,6);
case 6:if((U=pa.i)&&"error"in U&&(null==(rb=U)?0:null==(Va=rb.error)?0:Va.details))for(na=U.error.details,I=p(na),ma=I.next();!ma.done;ma=I.next())ea=ma.value,(Ee=ea["@type"])&&-1<Fu.indexOf(Ee)&&(delete ea["@type"],U=ea);t&&a.h.has(t)&&a.h.delete(t);(null==(Fe=b.config)?0:Fe.kd)&&Z(b.config.kd);if(U||null==(sd=a.i)||!sd.pr(b.input,b.ga)){pa.u(7);break}return v(pa,a.i.wr(b.input,b.ga),8);case 8:U=pa.i;case 7:return Mu(a,U,b),pa.return(U||void 0)}})}
function Ku(a,b){var c,d,e,f;return x(function(g){if(1==g.h){e=null==(c=a)?void 0:null==(d=c.ab)?void 0:d.sessionIndex;var h=ct({sessionIndex:e});if(!(h instanceof Af)){var k=new Af(cb);Bf(k,2,h);h=k}return v(g,h,2)}f=g.i;return g.return(Promise.resolve(Object.assign({},Ju(b),f)))})}
function Ju(a){var b={"Content-Type":"application/json"};L("EOM_VISITOR_DATA")?b["X-Goog-EOM-Visitor-Id"]=L("EOM_VISITOR_DATA"):L("VISITOR_DATA")&&(b["X-Goog-Visitor-Id"]=L("VISITOR_DATA"));M("track_webfe_innertube_auth_mismatch")&&(b["X-Youtube-Bootstrap-Logged-In"]=L("LOGGED_IN",!1));"cors"!==a&&((a=L("INNERTUBE_CONTEXT_CLIENT_NAME"))&&(b["X-Youtube-Client-Name"]=a),(a=L("INNERTUBE_CONTEXT_CLIENT_VERSION"))&&(b["X-Youtube-Client-Version"]=a),(a=L("CHROME_CONNECTED_HEADER"))&&(b["X-Youtube-Chrome-Connected"]=
a),(a=L("DOMAIN_ADMIN_STATE"))&&(b["X-Youtube-Domain-Admin-State"]=a));return b}
;var Nu=new Fq("INNERTUBE_TRANSPORT_TOKEN");var Ou=["share/get_web_player_share_panel"],Pu=["feedback"],Qu=["notification/modify_channel_preference"],Ru=["browse/edit_playlist"],Su=["subscription/subscribe"],Tu=["subscription/unsubscribe"];function Uu(){}
u(Uu,st);Uu.prototype.j=function(){return Su};
Uu.prototype.h=function(a){return Rq(a,Ck)||void 0};
Uu.prototype.i=function(a,b,c){c=void 0===c?{}:c;b.channelIds&&(a.channelIds=b.channelIds);b.siloName&&(a.siloName=b.siloName);b.params&&(a.params=b.params);c.botguardResponse&&(a.botguardResponse=c.botguardResponse);c.feature&&(a.clientFeature=c.feature)};
fa.Object.defineProperties(Uu.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return!0}}});function Vu(){}
u(Vu,st);Vu.prototype.j=function(){return Tu};
Vu.prototype.h=function(a){return Rq(a,Bk)||void 0};
Vu.prototype.i=function(a,b){b.channelIds&&(a.channelIds=b.channelIds);b.siloName&&(a.siloName=b.siloName);b.params&&(a.params=b.params)};
fa.Object.defineProperties(Vu.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return!0}}});function Wu(){}
u(Wu,st);Wu.prototype.j=function(){return Pu};
Wu.prototype.h=function(a){return Rq(a,lj)||void 0};
Wu.prototype.i=function(a,b,c){a.feedbackTokens=[];b.feedbackToken&&a.feedbackTokens.push(b.feedbackToken);if(b=b.cpn||c.cpn)a.feedbackContext={cpn:b};a.isFeedbackTokenUnencrypted=!!c.is_feedback_token_unencrypted;a.shouldMerge=!1;c.extra_feedback_tokens&&(a.shouldMerge=!0,a.feedbackTokens=a.feedbackTokens.concat(c.extra_feedback_tokens))};
fa.Object.defineProperties(Wu.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return!0}}});function Xu(){}
u(Xu,st);Xu.prototype.j=function(){return Qu};
Xu.prototype.h=function(a){return Rq(a,Ak)||void 0};
Xu.prototype.i=function(a,b){b.params&&(a.params=b.params);b.secondaryParams&&(a.secondaryParams=b.secondaryParams)};function Yu(){}
u(Yu,st);Yu.prototype.j=function(){return Ru};
Yu.prototype.h=function(a){return Rq(a,fk)||void 0};
Yu.prototype.i=function(a,b){b.actions&&(a.actions=b.actions);b.params&&(a.params=b.params);b.playlistId&&(a.playlistId=b.playlistId)};function Zu(){}
u(Zu,st);Zu.prototype.j=function(){return Ou};
Zu.prototype.h=function(a){return Rq(a,ek)};
Zu.prototype.i=function(a,b,c){c=void 0===c?{}:c;b.serializedShareEntity&&(a.serializedSharedEntity=b.serializedShareEntity);c.includeListId&&(a.includeListId=!0)};var Hq=new Fq("NETWORK_SLI_TOKEN");function $u(a){this.h=a}
$u.prototype.fetch=function(a,b){var c=this,d,e;return x(function(f){c.h&&(d=lc(mc(5,Cc(a,"key")))||"/UNKNOWN_PATH",c.h.start(d));e=new window.Request(a,b);return M("web_fetch_promise_cleanup_killswitch")?f.return(Promise.resolve(fetch(e).then(function(g){return c.handleResponse(g)}).catch(function(g){js(g)}))):f.return(fetch(e).then(function(g){return c.handleResponse(g)}).catch(function(g){js(g)}))})};
$u.prototype.handleResponse=function(a){var b=a.text().then(function(c){return JSON.parse(c.replace(")]}'",""))});
a.redirected||a.ok?this.h&&this.h.success():(this.h&&this.h.tr(),b=b.then(function(c){js(new P("Error: API fetch failed",a.status,a.url,c));return Object.assign({},c,{errorMetadata:{status:a.status}})}));
return b};
$u[Eq]=[new Gq];var av=new Fq("NETWORK_MANAGER_TOKEN");var bv;function cv(a){uo.call(this,1,arguments);this.csn=a}
u(cv,uo);var Do=new vo("screen-created",cv),dv=[],fv=ev,gv=0;function hv(a,b,c,d,e,f,g){function h(){js(new P("newScreen() parent element does not have a VE - rootVe",b))}
var k=fv(),m=new us({veType:b,youtubeData:f,jspbYoutubeData:void 0});f={Z:k};e&&(f.cttAuthInfo=e);M("il_via_jspb")?(e=new Cj,e.i(k),Dj(e,m.getAsJspb()),c&&c.visualElement?(m=new Ej,c.clientScreenNonce&&D(m,2,c.clientScreenNonce),Fj(m,c.visualElement.getAsJspb()),g&&D(m,4,Yj[g]),G(e,Ej,5,m)):c&&h(),d&&D(e,3,d),Or(e,f,a)):(e={csn:k,pageVe:m.getAsJson()},c&&c.visualElement?(e.implicitGesture={parentCsn:c.clientScreenNonce,gesturedVe:c.visualElement.getAsJson()},g&&(e.implicitGesture.gestureType=g)):
c&&h(),d&&(e.cloneCsn=d),a?Dr("screenCreated",e,a,f):jm("screenCreated",e,f));Ao(Do,new cv(k));return k}
function iv(a,b,c,d){var e=d.filter(function(k){k.csn!==b?(k.csn=b,k=!0):k=!1;return k}),f={cttAuthInfo:Ds(b)||void 0,
Z:b};d=p(d);for(var g=d.next();!g.done;g=d.next())g=g.value.getAsJson(),(pb(g)||!g.trackingParams&&!g.veType)&&js(Error("Child VE logged with no data"));if(M("il_via_jspb")){var h=new Gj;h.i(b);Ij(h,c.getAsJspb());hb(e,function(k){k=k.getAsJspb();ce(h,3,Aj,k)});
"UNDEFINED_CSN"===b?jv("visualElementAttached",f,void 0,h):Pr(h,f,a)}else c={csn:b,parentVe:c.getAsJson(),childVes:hb(e,function(k){return k.getAsJson()})},"UNDEFINED_CSN"===b?jv("visualElementAttached",f,c):a?Dr("visualElementAttached",c,a,f):jm("visualElementAttached",c,f)}
function ev(){for(var a=Math.random()+"",b=[],c=0,d=0;d<a.length;d++){var e=a.charCodeAt(d);255<e&&(b[c++]=e&255,e>>=8);b[c++]=e}return ad(b,3)}
function jv(a,b,c,d){dv.push({Eb:a,payload:c,ma:d,options:b});gv||(gv=Eo())}
function Fo(a){if(dv){for(var b=p(dv),c=b.next();!c.done;c=b.next())if(c=c.value,M("il_via_jspb")&&c.ma)switch(c.ma.i(a.csn),c.Eb){case "screenCreated":Or(c.ma,c.options);break;case "visualElementAttached":Pr(c.ma,c.options);break;case "visualElementShown":Kr(c.ma,c.options);break;case "visualElementHidden":Lr(c.ma,c.options);break;case "visualElementGestured":Mr(c.ma,c.options);break;case "visualElementStateChanged":Nr(c.ma,c.options);break;default:js(new P("flushQueue unable to map payloadName to JSPB setter"))}else c.payload&&
(c.payload.csn=a.csn,Dr(c.Eb,c.payload,null,c.options));dv.length=0}gv=0}
;function kv(){this.l=new Set;this.h=new Set;this.m=new Map;this.client=sp;this.csn=null}
function lv(){kv.h||(kv.h=new kv);return kv.h}
kv.prototype.i=function(a){this.client=a};
kv.prototype.j=function(){this.clear();this.csn=Bs()};
kv.prototype.clear=function(){this.l.clear();this.h.clear();this.m.clear();this.csn=null};function mv(){this.j=new Set;this.h=new Set;this.l=new Map;this.client=sp;this.csn=null}
function nv(){mv.h||(mv.h=new mv);return mv.h}
mv.prototype.i=function(a){M("safe_logging_library_killswitch")?this.client=a:Qk(lv().i).bind(lv())(a)};
mv.prototype.clear=function(){M("safe_logging_library_killswitch")?(this.j.clear(),this.h.clear(),this.l.clear(),this.csn=null):Qk(lv().clear).bind(lv())()};function ov(){this.j=new Set;this.h=new Set;this.l=new Map}
ov.prototype.i=function(a){M("use_ts_visibilitylogger")&&nv().i(a)};
ov.prototype.clear=function(){M("use_ts_visibilitylogger")?nv().clear():(this.j.clear(),this.h.clear(),this.l.clear())};
Na(ov);function pv(){this.o=[];this.v=[];this.h=[];this.m=[];this.M=[];this.j=new Set;this.s=new Map}
pv.prototype.i=function(a){this.client=a};
function qv(a,b,c){c=void 0===c?0:c;b.then(function(d){a.j.has(c)&&a.l&&a.l();var e=Bs(c),f=zs(c);if(e&&f){var g;(null==d?0:null==(g=d.response)?0:g.trackingParams)&&iv(a.client,e,f,[vs(d.response.trackingParams)]);var h;(null==d?0:null==(h=d.playerResponse)?0:h.trackingParams)&&iv(a.client,e,f,[vs(d.playerResponse.trackingParams)])}})}
function rv(a,b,c,d){d=void 0===d?0:d;if(a.j.has(d))a.o.push([b,c]);else{var e=Bs(d);c=c||zs(d);e&&c&&iv(a.client,e,c,[b])}}
pv.prototype.clickCommand=function(a,b,c){var d=a.clickTrackingParams;c=void 0===c?0:c;if(d)if(c=Bs(void 0===c?0:c)){a=this.client;var e=vs(d);d={cttAuthInfo:Ds(c)||void 0,Z:c};M("il_via_jspb")?(b=new Jj,b.i(c),e=e.getAsJspb(),G(b,Aj,2,e),D(b,4,Yj.INTERACTION_LOGGING_GESTURE_TYPE_GENERIC_CLICK),"UNDEFINED_CSN"===c?jv("visualElementGestured",d,void 0,b):Mr(b,d,a)):(e={csn:c,ve:e.getAsJson(),gestureType:"INTERACTION_LOGGING_GESTURE_TYPE_GENERIC_CLICK"},b&&(e.clientData=b),"UNDEFINED_CSN"===c?jv("visualElementGestured",
d,e):a?Dr("visualElementGestured",e,a,d):jm("visualElementGestured",e,d));b=!0}else b=!1;else b=!1;return b};
pv.prototype.visualElementStateChanged=function(a,b,c){c=void 0===c?0:c;0===c&&this.j.has(c)?this.v.push([a,b]):sv(this,a,b,c)};
function sv(a,b,c,d){d=void 0===d?0:d;var e=Bs(d);d=b||zs(d);e&&d&&(a=a.client,b={cttAuthInfo:Ds(e)||void 0,Z:e},M("il_via_jspb")?(c=new Mj,c.i(e),d=d.getAsJspb(),G(c,Aj,2,d),"UNDEFINED_CSN"===e?jv("visualElementStateChanged",b,void 0,c):Nr(c,b,a)):(c={csn:e,ve:d.getAsJson(),clientData:c},"UNDEFINED_CSN"===e?jv("visualElementStateChanged",b,c):a?Dr("visualElementStateChanged",c,a,b):jm("visualElementStateChanged",c,b)))}
function tv(a,b,c){c=void 0===c?{}:c;a.j.add(c.layer||0);a.l=function(){uv(a,b,c);var f=zs(c.layer);if(f){for(var g=p(a.o),h=g.next();!h.done;h=g.next())h=h.value,rv(a,h[0],h[1]||f,c.layer);f=p(a.v);for(g=f.next();!g.done;g=f.next())g=g.value,sv(a,g[0],g[1])}};
Bs(c.layer)||a.l();if(c.Wb)for(var d=p(c.Wb),e=d.next();!e.done;e=d.next())qv(a,e.value,c.layer);else is(Error("Delayed screen needs a data promise."))}
function uv(a,b,c){c=void 0===c?{}:c;c.layer||(c.layer=0);var d=void 0!==c.dd?c.dd:c.layer;var e=Bs(d);d=zs(d);var f;d&&(void 0!==c.parentCsn?f={clientScreenNonce:c.parentCsn,visualElement:d}:e&&"UNDEFINED_CSN"!==e&&(f={clientScreenNonce:e,visualElement:d}));var g,h=L("EVENT_ID");"UNDEFINED_CSN"===e&&h&&(g={servletData:{serializedServletEventId:h}});try{var k=hv(a.client,b,f,c.Vb,c.cttAuthInfo,g,c.ur)}catch(m){ls(m,{Er:b,rootVe:d,Br:void 0,qr:e,Ar:f,Vb:c.Vb});is(m);return}Es(k,b,c.layer,c.cttAuthInfo);
if(b=e&&"UNDEFINED_CSN"!==e&&d){a:{b=p(Object.values(ts));for(f=b.next();!f.done;f=b.next())if(Bs(f.value)===e){b=!0;break a}b=!1}b=!b}b&&(b=a.client,g=!0,h=(g=void 0===g?!1:g)?16:8,f={cttAuthInfo:Ds(e)||void 0,Z:e,Xb:g},M("il_via_jspb")?(h=new Kj,h.i(e),d=d.getAsJspb(),G(h,Aj,2,d),D(h,4,g?16:8),"UNDEFINED_CSN"===e?jv("visualElementHidden",f,void 0,h):Lr(h,f,b)):(d={csn:e,ve:d.getAsJson(),eventType:h},"UNDEFINED_CSN"===e?jv("visualElementHidden",f,d):b?Dr("visualElementHidden",d,b,f):jm("visualElementHidden",
d,f)));a.h[a.h.length-1]&&!a.h[a.h.length-1].csn&&(a.h[a.h.length-1].csn=k||"");zu({clientScreenNonce:k});d=ov.getInstance();M("use_ts_visibilitylogger")?(d=nv(),M("safe_logging_library_killswitch")?(d.clear(),d.csn=Bs()):Qk(lv().j).bind(lv())()):d.clear();d=zs(c.layer);e&&"UNDEFINED_CSN"!==e&&d&&(M("web_mark_root_visible")||M("music_web_mark_root_visible"))&&(e=k,k={cttAuthInfo:Ds(e)||void 0,Z:e},M("il_via_jspb")?(b=new Lj,b.i(e),f=d.getAsJspb(),G(b,Aj,2,f),D(b,4,1),"UNDEFINED_CSN"===e?jv("visualElementShown",
k,void 0,b):Kr(b,k)):(b={csn:e,ve:d.getAsJson(),eventType:1},"UNDEFINED_CSN"===e?jv("visualElementShown",k,b):jm("visualElementShown",b,k)));a.j.delete(c.layer||0);a.l=void 0;e=p(a.s);for(k=e.next();!k.done;k=e.next())b=p(k.value),k=b.next().value,b=b.next().value,b.has(c.layer)&&d&&rv(a,k,d,c.layer);for(c=0;c<a.m.length;c++){e=a.m[c];try{e()}catch(m){is(m)}}for(c=a.m.length=0;c<a.M.length;c++){e=a.M[c];try{e()}catch(m){is(m)}}}
;function vv(){var a,b,c;return x(function(d){if(1==d.h)return a=Mq().resolve(Nu),a?v(d,Iu(a),2):(js(Error("InnertubeTransportService unavailable in fetchDatasyncIds")),d.return(void 0));if(b=d.i){if(b.errorMetadata)return js(Error("Datasync IDs fetch responded with "+b.errorMetadata.status+": "+b.error)),d.return(void 0);c=b.rr;return d.return(c)}js(Error("Network request to get Datasync IDs failed."));return d.return(void 0)})}
;var wv=z.caches,xv;function yv(a){var b=a.indexOf(":");return-1===b?{jc:a}:{jc:a.substring(0,b),datasyncId:a.substring(b+1)}}
function zv(){return x(function(a){if(void 0!==xv)return a.return(xv);xv=new Promise(function(b){var c;return x(function(d){switch(d.h){case 1:return ya(d,2),v(d,wv.open("test-only"),4);case 4:return v(d,wv.delete("test-only"),5);case 5:za(d,3);break;case 2:if(c=Aa(d),c instanceof Error&&"SecurityError"===c.name)return b(!1),d.return();case 3:b("caches"in window),d.h=0}})});
return a.return(xv)})}
function Av(a){var b,c,d,e,f,g,h;x(function(k){if(1==k.h)return v(k,zv(),2);if(3!=k.h){if(!k.i)return k.return(!1);b=[];return v(k,wv.keys(),3)}c=k.i;d=p(c);for(e=d.next();!e.done;e=d.next())f=e.value,g=yv(f),h=g.datasyncId,!h||a.includes(h)||b.push(wv.delete(f));return k.return(Promise.all(b).then(function(m){return m.some(function(q){return q})}))})}
function Bv(){var a,b,c,d,e,f,g;return x(function(h){if(1==h.h)return v(h,zv(),2);if(3!=h.h){if(!h.i)return h.return(!1);a=Yl("cache contains other");return v(h,wv.keys(),3)}b=h.i;c=p(b);for(d=c.next();!d.done;d=c.next())if(e=d.value,f=yv(e),(g=f.datasyncId)&&g!==a)return h.return(!0);return h.return(!1)})}
;function Cv(){try{return!!self.localStorage}catch(a){return!1}}
;function Dv(a){a=a.match(/(.*)::.*::.*/);if(null!==a)return a[1]}
function Ev(a){if(Cv()){var b=Object.keys(window.localStorage);b=p(b);for(var c=b.next();!c.done;c=b.next()){c=c.value;var d=Dv(c);void 0===d||a.includes(d)||self.localStorage.removeItem(c)}}}
function Fv(){if(!Cv())return!1;var a=Yl(),b=Object.keys(window.localStorage);b=p(b);for(var c=b.next();!c.done;c=b.next())if(c=Dv(c.value),void 0!==c&&c!==a)return!0;return!1}
;function Gv(){vv().then(function(a){a&&(yn(a),Av(a),Ev(a))})}
function Hv(){var a=new gp;$h.R(function(){var b,c,d,e;return x(function(f){switch(f.h){case 1:if(M("ytidb_clear_optimizations_killswitch")){f.u(2);break}b=Yl("clear");if(b.startsWith("V")){var g=[b];yn(g);Av(g);Ev(g);return f.return()}c=Fv();return v(f,Bv(),3);case 3:return d=f.i,v(f,zn(),4);case 4:if(e=f.i,!c&&!d&&!e)return f.return();case 2:a.U()?Gv():a.l.add("publicytnetworkstatus-online",Gv,!0,void 0,void 0),f.h=0}})})}
;function Iv(a){a&&(a.dataset?a.dataset[Jv("loaded")]="true":a.setAttribute("data-loaded","true"))}
function Kv(a,b){return a?a.dataset?a.dataset[Jv(b)]:a.getAttribute("data-"+b):null}
var Lv={};function Jv(a){return Lv[a]||(Lv[a]=String(a).replace(/\-([a-z])/g,function(b,c){return c.toUpperCase()}))}
;var Mv=/\.vflset|-vfl[a-zA-Z0-9_+=-]+/,Nv=/-[a-zA-Z]{2,3}_[a-zA-Z]{2,3}(?=(\/|$))/;function Ov(a,b,c){c=void 0===c?null:c;if(window.spf&&spf.script){c="";if(a){var d=a.indexOf("jsbin/"),e=a.lastIndexOf(".js"),f=d+6;-1<d&&-1<e&&e>f&&(c=a.substring(f,e),c=c.replace(Mv,""),c=c.replace(Nv,""),c=c.replace("debug-",""),c=c.replace("tracing-",""))}spf.script.load(a,c,b)}else Pv(a,b,c)}
function Pv(a,b,c){c=void 0===c?null:c;var d=Qv(a),e=document.getElementById(d),f=e&&Kv(e,"loaded"),g=e&&!f;f?b&&b():(b&&(f=yq(d,b),b=""+Ra(b),Rv[b]=f),g||(e=Sv(a,d,function(){Kv(e,"loaded")||(Iv(e),Bq(d),kl(Ya(Cq,d),0))},c)))}
function Sv(a,b,c,d){d=void 0===d?null:d;var e=nf("SCRIPT");e.id=b;e.onload=function(){c&&setTimeout(c,0)};
e.onreadystatechange=function(){switch(e.readyState){case "loaded":case "complete":e.onload()}};
d&&e.setAttribute("nonce",d);Jh(e,Ib(a));a=document.getElementsByTagName("head")[0]||document.body;a.insertBefore(e,a.firstChild);return e}
function Tv(a){a=Qv(a);var b=document.getElementById(a);b&&(Cq(a),b.parentNode.removeChild(b))}
function Uv(a,b){a&&b&&(a=""+Ra(b),(a=Rv[a])&&Aq(a))}
function Qv(a){var b=document.createElement("a");ec(b,a);a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"js-"+jc(a)}
var Rv={};var Vv=[],Wv=!1;function Xv(){if(!M("disable_biscotti_fetch_for_ad_blocker_detection")&&!M("disable_biscotti_fetch_entirely_for_all_web_clients")&&Qs()){var a=L("PLAYER_VARS",{});if("1"!=sb(a)&&!Rs(a)){var b=function(){Wv=!0;"google_ad_status"in window?Ik("DCLKSTAT",1):Ik("DCLKSTAT",2)};
try{Ov("//static.doubleclick.net/instream/ad_status.js",b)}catch(c){}Vv.push($h.R(function(){if(!(Wv||"google_ad_status"in window)){try{Uv("//static.doubleclick.net/instream/ad_status.js",b)}catch(c){}Wv=!0;Ik("DCLKSTAT",3)}},5E3))}}}
function Yv(){var a=Number(L("DCLKSTAT",0));return isNaN(a)?0:a}
;function Zv(a){var b=this;var c=void 0===c?0:c;var d=void 0===d?Wl():d;this.l=c;this.j=d;this.i=new Eh;this.h=a;a={};c=p(this.h.entries());for(d=c.next();!d.done;a={Fa:a.Fa,Sa:a.Sa},d=c.next()){var e=p(d.value);d=e.next().value;e=e.next().value;a.Sa=d;a.Fa=e;d=function(f){return function(){f.Fa.Ab();b.h[f.Sa].mb=!0;b.h.every(function(g){return!0===g.mb})&&b.i.resolve()}}(a);
e=Sl(d,$v(this,a.Fa));this.h[a.Sa]=Object.assign({},a.Fa,{Ab:d,jobId:e})}}
function aw(a){var b=Array.from(a.h.keys()).sort(function(d,e){return $v(a,a.h[e])-$v(a,a.h[d])});
b=p(b);for(var c=b.next();!c.done;c=b.next())c=a.h[c.value],void 0===c.jobId||c.mb||(a.j.ea(c.jobId),Sl(c.Ab,10))}
Zv.prototype.cancel=function(){for(var a=p(this.h),b=a.next();!b.done;b=a.next())b=b.value,void 0===b.jobId||b.mb||this.j.ea(b.jobId),b.mb=!0;this.i.resolve()};
function $v(a,b){var c;return null!=(c=b.priority)?c:a.l}
;function bw(a){this.state=a;this.plugins=[];this.o=void 0}
bw.prototype.install=function(){this.plugins.push.apply(this.plugins,ia(Ja.apply(0,arguments)))};
bw.prototype.uninstall=function(){var a=this;Ja.apply(0,arguments).forEach(function(b){b=a.plugins.indexOf(b);-1<b&&a.plugins.splice(b,1)})};
bw.prototype.transition=function(a,b){var c=this,d=this.v.find(function(f){return Array.isArray(f.from)?f.from.find(function(g){return g===c.state&&f.D===a}):f.from===c.state&&f.D===a});
if(d){this.j&&(aw(this.j),this.j=void 0);this.state=a;d=d.action.bind(this);var e=this.plugins.filter(function(f){return f[a]}).map(function(f){return f[a]});
d(ew(this,e,this.o),b)}else throw Error("no transition specified from "+this.state+" to "+a);};
function ew(a,b,c){return function(){var d=Ja.apply(0,arguments),e=b.filter(function(k){var m;return 10===(null!=(m=null!=c?c:k.priority)?m:0)}),f=b.filter(function(k){var m;
return 10!==(null!=(m=null!=c?c:k.priority)?m:0)});
Wl();var g={};e=p(e);for(var h=e.next();!h.done;g={Ta:g.Ta},h=e.next())g.Ta=h.value,Ul(function(k){return function(){k.Ta.callback.apply(k.Ta,ia(d))}}(g));
f=f.map(function(k){var m;return{Ab:function(){k.callback.apply(k,ia(d))},
priority:null!=(m=null!=c?c:k.priority)?m:0}});
f.length&&(a.j=new Zv(f))}}
fa.Object.defineProperties(bw.prototype,{currentState:{configurable:!0,enumerable:!0,get:function(){return this.state}}});function fw(a){bw.call(this,void 0===a?"document_active":a);var b=this;this.o=10;this.h=new Map;this.v=[{from:"document_active",D:"document_disposed_preventable",action:this.M},{from:"document_active",D:"document_disposed",action:this.l},{from:"document_disposed_preventable",D:"document_disposed",action:this.l},{from:"document_disposed_preventable",D:"flush_logs",action:this.m},{from:"document_disposed_preventable",D:"document_active",action:this.i},{from:"document_disposed",D:"flush_logs",action:this.m},
{from:"document_disposed",D:"document_active",action:this.i},{from:"document_disposed",D:"document_disposed",action:function(){}},
{from:"flush_logs",D:"document_active",action:this.i}];window.addEventListener("pagehide",function(c){b.transition("document_disposed",{event:c})});
window.addEventListener("beforeunload",function(c){b.transition("document_disposed_preventable",{event:c})})}
u(fw,bw);fw.prototype.M=function(a,b){if(!this.h.get("document_disposed_preventable")){a(null==b?void 0:b.event);var c,d;if((null==b?0:null==(c=b.event)?0:c.defaultPrevented)||(null==b?0:null==(d=b.event)?0:d.returnValue)){b.event.returnValue||(b.event.returnValue=!0);b.event.defaultPrevented||b.event.preventDefault();this.h=new Map;this.transition("document_active");return}}this.h.set("document_disposed_preventable",!0);this.h.get("document_disposed")?this.transition("flush_logs"):this.transition("document_disposed")};
fw.prototype.l=function(a,b){this.h.get("document_disposed")?this.transition("document_active"):(a(null==b?void 0:b.event),this.h.set("document_disposed",!0),this.transition("flush_logs"))};
fw.prototype.m=function(a,b){a(null==b?void 0:b.event);this.transition("document_active")};
fw.prototype.i=function(){this.h=new Map};function gw(a){bw.call(this,void 0===a?"document_visibility_unknown":a);var b=this;this.v=[{from:"document_visibility_unknown",D:"document_visible",action:this.i},{from:"document_visibility_unknown",D:"document_hidden",action:this.h},{from:"document_visibility_unknown",D:"document_foregrounded",action:this.m},{from:"document_visibility_unknown",D:"document_backgrounded",action:this.l},{from:"document_visible",D:"document_hidden",action:this.h},{from:"document_visible",D:"document_foregrounded",action:this.m},
{from:"document_visible",D:"document_visible",action:this.i},{from:"document_foregrounded",D:"document_visible",action:this.i},{from:"document_foregrounded",D:"document_hidden",action:this.h},{from:"document_foregrounded",D:"document_foregrounded",action:this.m},{from:"document_hidden",D:"document_visible",action:this.i},{from:"document_hidden",D:"document_backgrounded",action:this.l},{from:"document_hidden",D:"document_hidden",action:this.h},{from:"document_backgrounded",D:"document_hidden",action:this.h},
{from:"document_backgrounded",D:"document_backgrounded",action:this.l},{from:"document_backgrounded",D:"document_visible",action:this.i}];document.addEventListener("visibilitychange",function(c){"visible"===document.visibilityState?b.transition("document_visible",{event:c}):b.transition("document_hidden",{event:c})});
M("visibility_lifecycles_dynamic_backgrounding")&&(window.addEventListener("blur",function(c){b.transition("document_backgrounded",{event:c})}),window.addEventListener("focus",function(c){b.transition("document_foregrounded",{event:c})}))}
u(gw,bw);gw.prototype.i=function(a,b){a(null==b?void 0:b.event);M("visibility_lifecycles_dynamic_backgrounding")&&this.transition("document_foregrounded")};
gw.prototype.h=function(a,b){a(null==b?void 0:b.event);M("visibility_lifecycles_dynamic_backgrounding")&&this.transition("document_backgrounded")};
gw.prototype.l=function(a,b){a(null==b?void 0:b.event)};
gw.prototype.m=function(a,b){a(null==b?void 0:b.event)};function hw(){this.h=new fw;this.i=new gw}
hw.prototype.install=function(){var a=Ja.apply(0,arguments);this.h.install.apply(this.h,ia(a));this.i.install.apply(this.i,ia(a))};function iw(){hw.call(this);var a={};this.install((a.document_disposed={callback:this.j},a));a={};this.install((a.flush_logs={callback:this.l},a))}
var jw;u(iw,hw);iw.prototype.l=function(){if(M("web_fp_via_jspb")){var a=new zj,b=Bs();b&&D(a,1,b);b=new Wj;ae(b,zj,380,Xj,a);Jr(b);M("web_fp_via_jspb_and_json")&&jm("finalPayload",{csn:Bs()})}else jm("finalPayload",{csn:Bs()})};
iw.prototype.j=function(){ns(os)};function kw(){}
kw.getInstance=function(){var a=B("ytglobal.storage_");a||(a=new kw,A("ytglobal.storage_",a));return a};
kw.prototype.estimate=function(){var a,b,c;return x(function(d){a=navigator;return(null==(b=a.storage)?0:b.estimate)?d.return(a.storage.estimate()):(null==(c=a.webkitTemporaryStorage)?0:c.queryUsageAndQuota)?d.return(lw()):d.return()})};
function lw(){var a=navigator;return new Promise(function(b,c){var d;null!=(d=a.webkitTemporaryStorage)&&d.queryUsageAndQuota?a.webkitTemporaryStorage.queryUsageAndQuota(function(e,f){b({usage:e,quota:f})},function(e){c(e)}):c(Error("webkitTemporaryStorage is not supported."))})}
A("ytglobal.storageClass_",kw);function hm(a,b){var c=this;this.handleError=a;this.h=b;this.i=!1;void 0===self.document||self.addEventListener("beforeunload",function(){c.i=!0});
this.j=Math.random()<=Lk("ytidb_transaction_ended_event_rate_limit_session",.2)}
hm.prototype.logEvent=function(a,b){switch(a){case "IDB_DATA_CORRUPTED":M("idb_data_corrupted_killswitch")||this.h("idbDataCorrupted",b);break;case "IDB_UNEXPECTEDLY_CLOSED":this.h("idbUnexpectedlyClosed",b);break;case "IS_SUPPORTED_COMPLETED":M("idb_is_supported_completed_killswitch")||this.h("idbIsSupportedCompleted",b);break;case "QUOTA_EXCEEDED":mw(this,b);break;case "TRANSACTION_ENDED":this.j&&Math.random()<=Lk("ytidb_transaction_ended_event_rate_limit_transaction",.1)&&this.h("idbTransactionEnded",
b);break;case "TRANSACTION_UNEXPECTEDLY_ABORTED":a=Object.assign({},b,{hasWindowUnloaded:this.i}),this.h("idbTransactionAborted",a)}};
function mw(a,b){kw.getInstance().estimate().then(function(c){c=Object.assign({},b,{isSw:void 0===self.document,isIframe:self!==self.top,deviceStorageUsageMbytes:nw(null==c?void 0:c.usage),deviceStorageQuotaMbytes:nw(null==c?void 0:c.quota)});a.h("idbQuotaExceeded",c)})}
function nw(a){return"undefined"===typeof a?"-1":String(Math.ceil(a/1048576))}
;function ow(a,b,c){J.call(this);var d=this;c=c||L("POST_MESSAGE_ORIGIN")||window.document.location.protocol+"//"+window.document.location.hostname;this.l=b||null;this.targetOrigin="*";this.m=c;this.sessionId=null;this.i="widget";this.I=!!a;this.F=function(e){a:if(!("*"!=d.m&&e.origin!=d.m||d.l&&e.source!=d.l||"string"!==typeof e.data)){try{var f=JSON.parse(e.data)}catch(g){break a}if(!(null==f||d.I&&(d.sessionId&&d.sessionId!=f.id||d.i&&d.i!=f.channel))&&f)switch(f.event){case "listening":"null"!=
e.origin&&(d.m=d.targetOrigin=e.origin);d.l=e.source;d.sessionId=f.id;d.j&&(d.j(),d.j=null);break;case "command":d.o&&(!d.s||0<=eb(d.s,f.func))&&d.o(f.func,f.args,e.origin)}}};
this.s=this.j=this.o=null;window.addEventListener("message",this.F)}
u(ow,J);ow.prototype.sendMessage=function(a,b){if(b=b||this.l){this.sessionId&&(a.id=this.sessionId);this.i&&(a.channel=this.i);try{var c=JSON.stringify(a);b.postMessage(c,this.targetOrigin)}catch(d){Sk(d)}}};
ow.prototype.B=function(){window.removeEventListener("message",this.F);J.prototype.B.call(this)};function pw(){this.i=[];this.isReady=!1;this.j={};var a=this.h=new ow(!!L("WIDGET_ID_ENFORCE")),b=this.fd.bind(this);a.o=b;a.s=null;this.h.i="widget";if(a=L("WIDGET_ID"))this.h.sessionId=a}
l=pw.prototype;l.fd=function(a,b,c){"addEventListener"===a&&b?(a=b[0],this.j[a]||"onReady"===a||(this.addEventListener(a,qw(this,a)),this.j[a]=!0)):this.Kb(a,b,c)};
l.Kb=function(){};
function qw(a,b){return function(c){return a.sendMessage(b,c)}}
l.addEventListener=function(){};
l.Qc=function(){this.isReady=!0;this.sendMessage("initialDelivery",this.wb());this.sendMessage("onReady");fb(this.i,this.qc,this);this.i=[]};
l.wb=function(){return null};
function rw(a,b){a.sendMessage("infoDelivery",b)}
l.qc=function(a){this.isReady?this.h.sendMessage(a):this.i.push(a)};
l.sendMessage=function(a,b){this.qc({event:a,info:void 0===b?null:b})};
l.dispose=function(){this.h=null};var sw={},tw=(sw["api.invalidparam"]=2,sw.auth=150,sw["drm.auth"]=150,sw["heartbeat.net"]=150,sw["heartbeat.servererror"]=150,sw["heartbeat.stop"]=150,sw["html5.unsupportedads"]=5,sw["fmt.noneavailable"]=5,sw["fmt.decode"]=5,sw["fmt.unplayable"]=5,sw["html5.missingapi"]=5,sw["html5.unsupportedlive"]=5,sw["drm.unavailable"]=5,sw["mrm.blocked"]=151,sw);var uw=new Set("endSeconds startSeconds mediaContentUrl suggestedQuality videoId rct rctn".split(" "));function vw(a){return(0===a.search("cue")||0===a.search("load"))&&"loadModule"!==a}
function ww(a,b,c){if("string"===typeof a)return{videoId:a,startSeconds:b,suggestedQuality:c};b={};c=p(uw);for(var d=c.next();!d.done;d=c.next())d=d.value,a[d]&&(b[d]=a[d]);return b}
function xw(a,b,c,d){if(Qa(a)&&!Array.isArray(a)){b="playlist list listType index startSeconds suggestedQuality".split(" ");c={};for(d=0;d<b.length;d++){var e=b[d];a[e]&&(c[e]=a[e])}return c}b={index:b,startSeconds:c,suggestedQuality:d};"string"===typeof a&&16===a.length?b.list="PL"+a:b.playlist=a;return b}
;function yw(a){pw.call(this);this.listeners=[];this.api=a;this.addEventListener("onReady",this.onReady.bind(this));this.addEventListener("onVideoProgress",this.qd.bind(this));this.addEventListener("onVolumeChange",this.rd.bind(this));this.addEventListener("onApiChange",this.ld.bind(this));this.addEventListener("onPlaybackQualityChange",this.nd.bind(this));this.addEventListener("onPlaybackRateChange",this.od.bind(this));this.addEventListener("onStateChange",this.pd.bind(this));this.addEventListener("onWebglSettingsChanged",
this.sd.bind(this))}
u(yw,pw);l=yw.prototype;
l.Kb=function(a,b,c){if(this.api.isExternalMethodAvailable(a,c)){b=b||[];if(0<b.length&&vw(a)){var d=b;if(Qa(d[0])&&!Array.isArray(d[0]))var e=d[0];else switch(e={},a){case "loadVideoById":case "cueVideoById":e=ww(d[0],void 0!==d[1]?Number(d[1]):void 0,d[2]);break;case "loadVideoByUrl":case "cueVideoByUrl":e=d[0];"string"===typeof e&&(e={mediaContentUrl:e,startSeconds:void 0!==d[1]?Number(d[1]):void 0,suggestedQuality:d[2]});b:{if((d=e.mediaContentUrl)&&(d=/\/([ve]|embed)\/([^#?]+)/.exec(d))&&d[2]){d=
d[2];break b}d=null}e.videoId=d;e=ww(e);break;case "loadPlaylist":case "cuePlaylist":e=xw(d[0],d[1],d[2],d[3])}b.length=1;b[0]=e}this.api.handleExternalCall(a,b,c);vw(a)&&rw(this,this.wb())}};
l.onReady=function(){var a=this.Qc.bind(this);this.h.j=a;a=this.api.getVideoData();if(!a.isPlayable){a=a.errorCode;var b=void 0===b?5:b;this.sendMessage("onError",(a?tw[a]||b:b).toString())}};
l.addEventListener=function(a,b){this.listeners.push({eventType:a,listener:b});this.api.addEventListener(a,b)};
l.wb=function(){if(!this.api)return null;var a=this.api.getApiInterface();kb(a,"getVideoData");for(var b={apiInterface:a},c=0,d=a.length;c<d;c++){var e=a[c];if(0===e.search("get")||0===e.search("is")){var f=0;0===e.search("get")?f=3:0===e.search("is")&&(f=2);f=e.charAt(f).toLowerCase()+e.substr(f+1);try{var g=this.api[e]();b[f]=g}catch(h){}}}b.videoData=this.api.getVideoData();b.currentTimeLastUpdated_=Date.now()/1E3;return b};
l.pd=function(a){a={playerState:a,currentTime:this.api.getCurrentTime(),duration:this.api.getDuration(),videoData:this.api.getVideoData(),videoStartBytes:0,videoBytesTotal:this.api.getVideoBytesTotal(),videoLoadedFraction:this.api.getVideoLoadedFraction(),playbackQuality:this.api.getPlaybackQuality(),availableQualityLevels:this.api.getAvailableQualityLevels(),currentTimeLastUpdated_:Date.now()/1E3,playbackRate:this.api.getPlaybackRate(),mediaReferenceTime:this.api.getMediaReferenceTime()};this.api.getVideoUrl&&
(a.videoUrl=this.api.getVideoUrl());this.api.getVideoContentRect&&(a.videoContentRect=this.api.getVideoContentRect());this.api.getProgressState&&(a.progressState=this.api.getProgressState());this.api.getPlaylist&&(a.playlist=this.api.getPlaylist());this.api.getPlaylistIndex&&(a.playlistIndex=this.api.getPlaylistIndex());this.api.getStoryboardFormat&&(a.storyboardFormat=this.api.getStoryboardFormat());rw(this,a)};
l.nd=function(a){rw(this,{playbackQuality:a})};
l.od=function(a){rw(this,{playbackRate:a})};
l.ld=function(){for(var a=this.api.getOptions(),b={namespaces:a},c=0,d=a.length;c<d;c++){var e=a[c],f=this.api.getOptions(e);b[e]={options:f};for(var g=0,h=f.length;g<h;g++){var k=f[g],m=this.api.getOption(e,k);b[e][k]=m}}this.sendMessage("apiInfoDelivery",b)};
l.rd=function(){rw(this,{muted:this.api.isMuted(),volume:this.api.getVolume()})};
l.qd=function(a){a={currentTime:a,videoBytesLoaded:this.api.getVideoBytesLoaded(),videoLoadedFraction:this.api.getVideoLoadedFraction(),currentTimeLastUpdated_:Date.now()/1E3,playbackRate:this.api.getPlaybackRate(),mediaReferenceTime:this.api.getMediaReferenceTime()};this.api.getProgressState&&(a.progressState=this.api.getProgressState());rw(this,a)};
l.sd=function(){var a={sphericalProperties:this.api.getSphericalProperties()};rw(this,a)};
l.dispose=function(){pw.prototype.dispose.call(this);for(var a=0;a<this.listeners.length;a++){var b=this.listeners[a];this.api.removeEventListener(b.eventType,b.listener)}this.listeners=[]};function zw(a){J.call(this);this.i={};this.started=!1;this.connection=a;this.connection.subscribe("command",this.lc,this)}
u(zw,J);l=zw.prototype;l.start=function(){this.started||this.h()||(this.started=!0,this.connection.ya("RECEIVING"))};
l.ya=function(a,b){this.started&&!this.h()&&this.connection.ya(a,b)};
l.lc=function(a,b,c){if(this.started&&!this.h()){var d=b||{};switch(a){case "addEventListener":"string"===typeof d.event&&this.addListener(d.event);break;case "removeEventListener":"string"===typeof d.event&&this.removeListener(d.event);break;default:this.api.isReady()&&this.api.isExternalMethodAvailable(a,c||null)&&(b=Aw(a,b||{}),c=this.api.handleExternalCall(a,b,c||null),(c=Bw(a,c))&&this.ya(a,c))}}};
l.addListener=function(a){if(!(a in this.i)){var b=this.md.bind(this,a);this.i[a]=b;this.addEventListener(a,b)}};
l.md=function(a,b){this.started&&!this.h()&&this.connection.ya(a,this.vb(a,b))};
l.vb=function(a,b){if(null!=b)return{value:b}};
l.removeListener=function(a){a in this.i&&(this.removeEventListener(a,this.i[a]),delete this.i[a])};
l.B=function(){var a=this.connection;a.h()||wi(a.i,"command",this.lc,this);this.connection=null;for(var b in this.i)this.i.hasOwnProperty(b)&&this.removeListener(b);J.prototype.B.call(this)};function Cw(a,b){zw.call(this,b);this.api=a;this.start()}
u(Cw,zw);Cw.prototype.addEventListener=function(a,b){this.api.addEventListener(a,b)};
Cw.prototype.removeEventListener=function(a,b){this.api.removeEventListener(a,b)};
function Aw(a,b){switch(a){case "loadVideoById":return a=ww(b),[a];case "cueVideoById":return a=ww(b),[a];case "loadVideoByPlayerVars":return[b];case "cueVideoByPlayerVars":return[b];case "loadPlaylist":return a=xw(b),[a];case "cuePlaylist":return a=xw(b),[a];case "seekTo":return[b.seconds,b.allowSeekAhead];case "playVideoAt":return[b.index];case "setVolume":return[b.volume];case "setPlaybackQuality":return[b.suggestedQuality];case "setPlaybackRate":return[b.suggestedRate];case "setLoop":return[b.loopPlaylists];
case "setShuffle":return[b.shufflePlaylist];case "getOptions":return[b.module];case "getOption":return[b.module,b.option];case "setOption":return[b.module,b.option,b.value];case "handleGlobalKeyDown":return[b.keyCode,b.shiftKey,b.ctrlKey,b.altKey,b.metaKey,b.key,b.code]}return[]}
function Bw(a,b){switch(a){case "isMuted":return{muted:b};case "getVolume":return{volume:b};case "getPlaybackRate":return{playbackRate:b};case "getAvailablePlaybackRates":return{availablePlaybackRates:b};case "getVideoLoadedFraction":return{videoLoadedFraction:b};case "getPlayerState":return{playerState:b};case "getCurrentTime":return{currentTime:b};case "getPlaybackQuality":return{playbackQuality:b};case "getAvailableQualityLevels":return{availableQualityLevels:b};case "getDuration":return{duration:b};
case "getVideoUrl":return{videoUrl:b};case "getVideoEmbedCode":return{videoEmbedCode:b};case "getPlaylist":return{playlist:b};case "getPlaylistIndex":return{playlistIndex:b};case "getOptions":return{options:b};case "getOption":return{option:b}}}
Cw.prototype.vb=function(a,b){switch(a){case "onReady":return;case "onStateChange":return{playerState:b};case "onPlaybackQualityChange":return{playbackQuality:b};case "onPlaybackRateChange":return{playbackRate:b};case "onError":return{errorCode:b}}return zw.prototype.vb.call(this,a,b)};
Cw.prototype.B=function(){zw.prototype.B.call(this);delete this.api};function Dw(a){a=void 0===a?!1:a;J.call(this);this.i=new K(a);ue(this,this.i)}
Za(Dw,J);Dw.prototype.subscribe=function(a,b,c){return this.h()?0:this.i.subscribe(a,b,c)};
Dw.prototype.m=function(a,b){this.h()||this.i.ra.apply(this.i,arguments)};function Ew(a,b,c){Dw.call(this);this.l=a;this.j=b;this.id=c}
u(Ew,Dw);Ew.prototype.ya=function(a,b){this.h()||this.l.ya(this.j,this.id,a,b)};
Ew.prototype.B=function(){this.j=this.l=null;Dw.prototype.B.call(this)};function Fw(a,b,c){J.call(this);this.i=a;this.origin=c;this.j=lq(window,"message",this.l.bind(this));this.connection=new Ew(this,a,b);ue(this,this.connection)}
u(Fw,J);Fw.prototype.ya=function(a,b,c,d){this.h()||a!==this.i||(a={id:b,command:c},d&&(a.data=d),this.i.postMessage(JSON.stringify(a),this.origin))};
Fw.prototype.l=function(a){if(!this.h()&&a.origin===this.origin){var b=a.data;if("string"===typeof b){try{b=JSON.parse(b)}catch(d){return}if(b.command){var c=this.connection;c.h()||c.m("command",b.command,b.data,a.origin)}}}};
Fw.prototype.B=function(){mq(this.j);this.i=null;J.prototype.B.call(this)};function Gw(){this.state=1;this.h=null}
l=Gw.prototype;
l.initialize=function(a,b,c){if(a.program){var d,e=null!=(d=a.interpreterScript)?d:null,f;d=null!=(f=a.interpreterUrl)?f:null;a.interpreterSafeScript&&(e=a.interpreterSafeScript,Cb("From proto message. b/166824318"),e=e.privateDoNotAccessOrElseSafeScriptWrappedValue||"",e=(f=zb())?f.createScript(e):e,e=(new Eb(e)).toString());a.interpreterSafeUrl&&(d=a.interpreterSafeUrl,Cb("From proto message. b/166824318"),d=Ib(d.privateDoNotAccessOrElseTrustedResourceUrlWrappedValue||"").toString());Hw(this,e,
d,a.program,b,c)}else js(Error("Cannot initialize botguard without program"))};
function Hw(a,b,c,d,e,f){var g=void 0===g?"trayride":g;c?(a.state=2,Ov(c,function(){window[g]?Iw(a,d,g,e):(a.state=3,Tv(c),js(new P("Unable to load Botguard","from "+c)))},f)):b?(f=nf("SCRIPT"),f.textContent=b,f.nonce=hc(),document.head.appendChild(f),document.head.removeChild(f),window[g]?Iw(a,d,g,e):(a.state=4,js(new P("Unable to load Botguard from JS")))):js(new P("Unable to load VM; no url or JS provided"))}
function Iw(a,b,c,d){a.state=5;try{var e=new Fh({program:b,Tc:c,gd:M("att_web_record_metrics")});e.td.then(function(){a.state=6;d&&d(b)});
a.Ib(e)}catch(f){a.state=7,f instanceof Error&&js(f)}}
l.invoke=function(a){a=void 0===a?{}:a;return this.Jb()?this.yc({Ub:a}):null};
l.dispose=function(){this.Lb()};
l.Lb=function(){this.Ib(null);this.state=8};
l.Jb=function(){return!!this.h};
l.yc=function(a){return this.h.tc(a)};
l.Ib=function(a){se(this.h);this.h=a};function Jw(){var a=B("yt.abuse.playerAttLoader");return a&&["bgvma","bgvmb","bgvmc"].every(function(b){return b in a})?a:null}
;function Kw(){Gw.apply(this,arguments)}
u(Kw,Gw);Kw.prototype.Lb=function(){this.state=8};
Kw.prototype.Ib=function(a){var b;null==(b=Jw())||b.bgvma();a?(b={bgvma:a.dispose.bind(a),bgvmb:a.snapshot.bind(a),bgvmc:a.tc.bind(a)},A("yt.abuse.playerAttLoader",b),A("yt.abuse.playerAttLoaderRun",function(c){return a.snapshot(c)})):(A("yt.abuse.playerAttLoader",null),A("yt.abuse.playerAttLoaderRun",null))};
Kw.prototype.Jb=function(){return!!Jw()};
Kw.prototype.yc=function(a){return Jw().bgvmc(a)};var Lw=new Kw;function Mw(){return Lw.Jb()}
function Nw(a){a=void 0===a?{}:a;return Lw.invoke(a)}
;function Ow(a){a=a||{};var b={},c={};this.url=a.url||"";this.args=a.args||ub(b);this.assets=a.assets||{};this.attrs=a.attrs||ub(c);this.fallback=a.fallback||null;this.fallbackMessage=a.fallbackMessage||null;this.html5=!!a.html5;this.disable=a.disable||{};this.loaded=!!a.loaded;this.messages=a.messages||{}}
Ow.prototype.clone=function(){var a=new Ow,b;for(b in this)if(this.hasOwnProperty(b)){var c=this[b];"object"==Oa(c)?a[b]=ub(c):a[b]=c}return a};var Pw=/cssbin\/(?:debug-)?([a-zA-Z0-9_-]+?)(?:-2x|-web|-rtl|-vfl|.css)/;function Qw(a){a=a||"";if(window.spf){var b=a.match(Pw);spf.style.load(a,b?b[1]:"",void 0)}else Rw(a)}
function Rw(a){var b=Sw(a),c=document.getElementById(b),d=c&&Kv(c,"loaded");d||c&&!d||(c=Tw(a,b,function(){Kv(c,"loaded")||(Iv(c),Bq(b),kl(Ya(Cq,b),0))}))}
function Tw(a,b,c){var d=document.createElement("link");d.id=b;d.onload=function(){c&&setTimeout(c,0)};
a=Ib(a);fc(d,a);(document.getElementsByTagName("head")[0]||document.body).appendChild(d);return d}
function Sw(a){var b=nf("A");Cb("This URL is never added to the DOM");ec(b,new Nb(a,Ob));a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"css-"+jc(a)}
;function Uw(){J.call(this);this.i=[]}
u(Uw,J);Uw.prototype.B=function(){for(;this.i.length;){var a=this.i.pop();a.target.removeEventListener(a.name,a.callback,void 0)}J.prototype.B.call(this)};function Vw(){Uw.apply(this,arguments)}
u(Vw,Uw);function Ww(a,b,c,d,e){J.call(this);var f=this;this.s=b;this.webPlayerContextConfig=d;this.Xa=e;this.ca=!1;this.api={};this.V=this.o=null;this.K=new K;this.i={};this.P=this.W=this.elementId=this.sa=this.config=null;this.O=!1;this.l=this.F=null;this.ja={};this.Ya=["onReady"];this.lastError=null;this.Ha=NaN;this.I={};this.Za=new Vw(this);this.S=0;this.j=this.m=a;ue(this,this.K);Xw(this);Yw(this);ue(this,this.Za);c?this.S=kl(function(){f.loadNewVideoConfig(c)},0):d&&(Zw(this),$w(this))}
u(Ww,J);l=Ww.prototype;l.getId=function(){return this.s};
l.loadNewVideoConfig=function(a){if(!this.h()){this.S&&(window.clearTimeout(this.S),this.S=0);var b=a||{};b instanceof Ow||(b=new Ow(b));this.config=b;this.setConfig(a);$w(this);this.isReady()&&ax(this)}};
function Zw(a){var b;a.webPlayerContextConfig?b=a.webPlayerContextConfig.rootElementId:b=a.config.attrs.id;a.elementId=b||a.elementId;"video-player"===a.elementId&&(a.elementId=a.s,a.webPlayerContextConfig?a.webPlayerContextConfig.rootElementId=a.s:a.config.attrs.id=a.s);var c;(null==(c=a.j)?void 0:c.id)===a.elementId&&(a.elementId+="-player",a.webPlayerContextConfig?a.webPlayerContextConfig.rootElementId=a.elementId:a.config.attrs.id=a.elementId)}
l.setConfig=function(a){this.sa=a;this.config=bx(a);Zw(this);if(!this.W){var b;this.W=cx(this,(null==(b=this.config.args)?void 0:b.jsapicallback)||"onYouTubePlayerReady")}this.config.args?this.config.args.jsapicallback=null:this.config.args={jsapicallback:null};var c;if(null==(c=this.config)?0:c.attrs)a=this.config.attrs,(b=a.width)&&this.j&&(this.j.style.width=Th(Number(b)||b)),(a=a.height)&&this.j&&(this.j.style.height=Th(Number(a)||a))};
function ax(a){if(a.config&&!0!==a.config.loaded)if(a.config.loaded=!0,!a.config.args||"0"!==a.config.args.autoplay&&0!==a.config.args.autoplay&&!1!==a.config.args.autoplay){var b;a.api.loadVideoByPlayerVars(null!=(b=a.config.args)?b:null)}else a.api.cueVideoByPlayerVars(a.config.args)}
function dx(a){var b=!0,c=ex(a);c&&a.config&&(a=fx(a),b=Kv(c,"version")===a);return b&&!!B("yt.player.Application.create")}
function $w(a){if(!a.h()&&!a.O){var b=dx(a);if(b&&"html5"===(ex(a)?"html5":null))a.P="html5",a.isReady()||gx(a);else if(hx(a),a.P="html5",b&&a.l&&a.m)a.m.appendChild(a.l),gx(a);else{a.config&&(a.config.loaded=!0);var c=!1;a.F=function(){c=!0;var d=ix(a,"player_bootstrap_method")?B("yt.player.Application.createAlternate")||B("yt.player.Application.create"):B("yt.player.Application.create");var e=a.config?bx(a.config):void 0;d&&d(a.m,e,a.webPlayerContextConfig,a.Xa);gx(a)};
a.O=!0;b?a.F():(Ov(fx(a),a.F),(b=jx(a))&&Qw(b),kx(a)&&!c&&A("yt.player.Application.create",null))}}}
function ex(a){var b=mf(a.elementId);!b&&a.j&&a.j.querySelector&&(b=a.j.querySelector("#"+a.elementId));return b}
function gx(a){if(!a.h()){var b=ex(a),c=!1;b&&b.getApiInterface&&b.getApiInterface()&&(c=!0);if(c){a.O=!1;if(!ix(a,"html5_remove_not_servable_check_killswitch")){var d;if((null==b?0:b.isNotServable)&&a.config&&(null==b?0:b.isNotServable(null==(d=a.config.args)?void 0:d.video_id)))return}lx(a)}else a.Ha=kl(function(){gx(a)},50)}}
function lx(a){Xw(a);a.ca=!0;var b=ex(a);if(b){a.o=mx(a,b,"addEventListener");a.V=mx(a,b,"removeEventListener");var c=b.getApiInterface();c=c.concat(b.getInternalApiInterface());for(var d=a.api,e=0;e<c.length;e++){var f=c[e];d[f]||(d[f]=mx(a,b,f))}}for(var g in a.i)a.i.hasOwnProperty(g)&&a.o&&a.o(g,a.i[g]);ax(a);a.W&&a.W(a.api);a.K.ra("onReady",a.api)}
function mx(a,b,c){var d=b[c];return function(){var e=Ja.apply(0,arguments);try{return a.lastError=null,d.apply(b,e)}catch(f){"sendAbandonmentPing"!==c&&(f.params=c,a.lastError=f,js(f))}}}
function Xw(a){a.ca=!1;if(a.V)for(var b in a.i)a.i.hasOwnProperty(b)&&a.V(b,a.i[b]);for(var c in a.I)a.I.hasOwnProperty(c)&&window.clearTimeout(Number(c));a.I={};a.o=null;a.V=null;b=a.api;for(var d in b)b.hasOwnProperty(d)&&(b[d]=null);b.addEventListener=function(e,f){a.addEventListener(e,f)};
b.removeEventListener=function(e,f){a.removeEventListener(e,f)};
b.destroy=function(){a.dispose()};
b.getLastError=function(){return a.getLastError()};
b.getPlayerType=function(){return a.getPlayerType()};
b.getCurrentVideoConfig=function(){return a.sa};
b.loadNewVideoConfig=function(e){a.loadNewVideoConfig(e)};
b.isReady=function(){return a.isReady()}}
l.isReady=function(){return this.ca};
function Yw(a){a.addEventListener("WATCH_LATER_VIDEO_ADDED",function(b){Bq("WATCH_LATER_VIDEO_ADDED",b)});
a.addEventListener("WATCH_LATER_VIDEO_REMOVED",function(b){Bq("WATCH_LATER_VIDEO_REMOVED",b)})}
l.addEventListener=function(a,b){var c=this,d=cx(this,b);d&&(0<=eb(this.Ya,a)||this.i[a]||(b=nx(this,a),this.o&&this.o(a,b)),this.K.subscribe(a,d),"onReady"===a&&this.isReady()&&kl(function(){d(c.api)},0))};
l.removeEventListener=function(a,b){this.h()||(b=cx(this,b))&&wi(this.K,a,b)};
function cx(a,b){var c=b;if("string"===typeof b){if(a.ja[b])return a.ja[b];c=function(){var d=Ja.apply(0,arguments),e=B(b);if(e)try{e.apply(z,d)}catch(f){is(f)}};
a.ja[b]=c}return c?c:null}
function nx(a,b){var c="ytPlayer"+b+a.s;a.i[b]=c;z[c]=function(d){var e=kl(function(){if(!a.h()){try{a.K.ra(b,null!=d?d:void 0)}catch(h){js(new P("PlayerProxy error when creating global callback",{error:h,event:b,playerId:a.s,data:d}))}var f=a.I,g=String(e);g in f&&delete f[g]}},0);
qb(a.I,String(e))};
return c}
l.getPlayerType=function(){return this.P||(ex(this)?"html5":null)};
l.getLastError=function(){return this.lastError};
function hx(a){a.cancel();Xw(a);a.P=null;a.config&&(a.config.loaded=!1);var b=ex(a);b&&(dx(a)||!kx(a)?a.l=b:(b&&b.destroy&&b.destroy(),a.l=null));if(a.m)for(a=a.m;b=a.firstChild;)a.removeChild(b)}
l.cancel=function(){this.F&&Uv(fx(this),this.F);window.clearTimeout(this.Ha);this.O=!1};
l.B=function(){hx(this);if(this.l&&this.config&&this.l.destroy)try{this.l.destroy()}catch(b){is(b)}this.ja=null;for(var a in this.i)this.i.hasOwnProperty(a)&&(z[this.i[a]]=null);this.sa=this.config=this.api=null;delete this.m;delete this.j;J.prototype.B.call(this)};
function kx(a){var b,c;a=null==(b=a.config)?void 0:null==(c=b.args)?void 0:c.fflags;return!!a&&-1!==a.indexOf("player_destroy_old_version=true")}
function fx(a){return a.webPlayerContextConfig?a.webPlayerContextConfig.jsUrl:(a=a.config.assets)?a.js:""}
function jx(a){return a.webPlayerContextConfig?a.webPlayerContextConfig.cssUrl:(a=a.config.assets)?a.css:""}
function ix(a,b){if(a.webPlayerContextConfig)var c=a.webPlayerContextConfig.serializedExperimentFlags;else{var d;if(null==(d=a.config)?0:d.args)c=a.config.args.fflags}return"true"===Yk(c||"","&")[b]}
function bx(a){for(var b={},c=p(Object.keys(a)),d=c.next();!d.done;d=c.next()){d=d.value;var e=a[d];b[d]="object"===typeof e?ub(e):e}return b}
;var ox={},px="player_uid_"+(1E9*Math.random()>>>0);function qx(a,b){var c="player",d=!1;d=void 0===d?!0:d;c="string"===typeof c?mf(c):c;var e=px+"_"+Ra(c),f=ox[e];if(f&&d)return rx(a,b)?f.api.loadVideoByPlayerVars(a.args||null):f.loadNewVideoConfig(a),f.api;f=new Ww(c,e,a,b,void 0);ox[e]=f;Bq("player-added",f.api);ve(f,function(){delete ox[f.getId()]});
return f.api}
function rx(a,b){return b&&b.serializedExperimentFlags?b.serializedExperimentFlags.includes("web_player_remove_playerproxy=true"):a&&a.args&&a.args.fflags?a.args.fflags.includes("web_player_remove_playerproxy=true"):!1}
;var sx=null,tx=null,ux=null;function vx(){wx()}
function xx(){wx()}
function wx(){var a=sx.getVideoData(1);a=a.title?a.title+" - YouTube":"YouTube";document.title!==a&&(document.title=a)}
function yx(){sx&&sx.sendAbandonmentPing&&sx.sendAbandonmentPing();L("PL_ATT")&&Lw.dispose();for(var a=$h,b=0,c=Vv.length;b<c;b++)a.ea(Vv[b]);Vv.length=0;Tv("//static.doubleclick.net/instream/ad_status.js");Wv=!1;Ik("DCLKSTAT",0);te(ux,tx);sx&&(sx.removeEventListener("onVideoDataChange",vx),sx.destroy())}
;function zx(a,b,c){a="ST-"+jc(a).toString(36);b=b?sc(b):"";c=c||5;Qs()&&Al(a,b,c)}
;function Ax(a,b,c){b=void 0===b?{}:b;c=void 0===c?!1:c;var d=L("EVENT_ID");d&&(b.ei||(b.ei=d));if(b){d=a;var e=void 0===e?!0:e;var f=L("VALID_SESSION_TEMPDATA_DOMAINS",[]),g=nc(window.location.href);g&&f.push(g);g=nc(d);if(0<=eb(f,g)||!g&&0==d.lastIndexOf("/",0))if(M("autoescape_tempdata_url")&&(f=document.createElement("a"),ec(f,d),d=f.href),d&&(d=oc(d),f=d.indexOf("#"),d=0>f?d:d.slice(0,f)))if(e&&!b.csn&&(b.itct||b.ved)&&(b=Object.assign({csn:Bs()},b)),h){var h=parseInt(h,10);isFinite(h)&&0<h&&
zx(d,b,h)}else zx(d,b)}if(c)return!1;if((window.ytspf||{}).enabled)spf.navigate(a);else{var k=void 0===k?{}:k;var m=void 0===m?"":m;var q=void 0===q?window:q;c=q.location;a=uc(a,k)+m;var r=void 0===r?Mh:r;a:{r=void 0===r?Mh:r;for(k=0;k<r.length;++k)if(m=r[k],m instanceof Kh&&m.Yc(a)){r=new Nb(a,Ob);break a}r=void 0}r=r||Rb;if(r instanceof Nb)var w=Pb(r);else{b:if(Ih){try{w=new URL(r)}catch(t){w="https:";break b}w=w.protocol}else c:{w=document.createElement("a");try{w.href=r}catch(t){w=void 0;break c}w=
w.protocol;w=":"===w||""===w?"https:":w}w="javascript:"!==w?r:void 0}void 0!==w&&(c.href=w)}return!0}
;A("yt.setConfig",Ik);A("yt.config.set",Ik);A("yt.setMsg",Gs);A("yt.msgs.set",Gs);A("yt.logging.errors.log",is);
A("writeEmbed",function(){var a=L("PLAYER_CONFIG");if(!a){var b=L("PLAYER_VARS");b&&(a={args:b})}$s(!0);"gvn"===a.args.ps&&(document.body.style.backgroundColor="transparent");a.attrs||(a.attrs={width:"100%",height:"100%",id:"video-player"});var c=document.referrer;b=L("POST_MESSAGE_ORIGIN");window!==window.top&&c&&c!==document.URL&&(a.args.loaderUrl=c);tu();c=L("WEB_PLAYER_CONTEXT_CONFIGS").WEB_PLAYER_CONTEXT_CONFIG_ID_EMBEDDED_PLAYER;if(!c.serializedForcedExperimentIds){var d=cl(window.location.href);
d.forced_experiments&&(c.serializedForcedExperimentIds=d.forced_experiments)}sx=qx(a,c);sx.addEventListener("onVideoDataChange",vx);sx.addEventListener("onReady",xx);a=L("POST_MESSAGE_ID","player");L("ENABLE_JS_API")?ux=new yw(sx):L("ENABLE_POST_API")&&"string"===typeof a&&"string"===typeof b&&(tx=new Fw(window.parent,a,b),ux=new Cw(sx,tx.connection));Xv();M("ytidb_create_logger_embed_killswitch")||M("embeds_web_disable_nwl")||gm();a={};jw||(jw=new iw);jw.install((a.flush_logs={callback:function(){lr()}},
a));
M("embeds_web_disable_nwl")||rp();M("ytidb_clear_embedded_player")&&$h.R(function(){var e;if(!bv){var f=Mq(),g={Fb:av,wc:$u};f.h.set(g.Fb,g);g={Tb:{feedbackEndpoint:ot(Wu),modifyChannelNotificationPreferenceEndpoint:ot(Xu),playlistEditEndpoint:ot(Yu),subscribeEndpoint:ot(Uu),unsubscribeEndpoint:ot(Vu),webPlayerShareEntityServiceEndpoint:ot(Zu)}};var h=kt.getInstance(),k={};h&&(k.client_location=h);if(void 0===m){bt.h||(bt.h=new bt);var m=bt.h}void 0===e&&(e=f.resolve(av));Hu(g,e,m,k);m={Fb:Nu,xc:Gu.h};
f.h.set(m.Fb,m);bv=f.resolve(Nu)}Hv()})});
var Bx=Qk(function(){Au();at();pv.h||(pv.h=new pv);var a=pv.h;var b=16623;var c=void 0===c?{}:c;Object.values(Hs).includes(b)||(js(new P("createClientScreen() called with a non-page VE",b)),b=83769);c.isHistoryNavigation||a.h.push({rootVe:b,key:c.key||""});a.o=[];a.v=[];c.Wb?tv(a,b,c):uv(a,b,c)}),Cx=Qk(function(a){M("embeds_web_enable_load_player_from_page_show")&&!a.persisted&&(Au(),at())}),Dx=Qk(function(a){M("embeds_web_enable_dispose_player_if_page_not_cached_killswitch")?yx():a.persisted||(M("embeds_web_enable_load_player_from_page_show")?
yx():Yc?setTimeout(function(){yx()},0):yx())}),Ex=Qk(yx);
window.addEventListener?(window.addEventListener("load",Bx),M("embeds_web_enable_load_player_from_page_show")?(window.addEventListener("pageshow",Cx),window.addEventListener("pagehide",Dx)):Yc?window.addEventListener("beforeunload",Dx):window.addEventListener("pagehide",Dx)):window.attachEvent&&(window.attachEvent("onload",Bx),window.attachEvent("onunload",Ex));A("yt.abuse.player.botguardInitialized",B("yt.abuse.player.botguardInitialized")||Mw);
A("yt.abuse.player.invokeBotguard",B("yt.abuse.player.invokeBotguard")||Nw);A("yt.abuse.dclkstatus.checkDclkStatus",B("yt.abuse.dclkstatus.checkDclkStatus")||Yv);A("yt.player.exports.navigate",B("yt.player.exports.navigate")||Ax);A("yt.util.activity.init",B("yt.util.activity.init")||qq);A("yt.util.activity.getTimeSinceActive",B("yt.util.activity.getTimeSinceActive")||tq);A("yt.util.activity.setTimestamp",B("yt.util.activity.setTimestamp")||rq);}).call(this);
