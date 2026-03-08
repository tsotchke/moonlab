declare namespace THREE {
  class Scene {}
  class WebGLRenderer {}
  class Points {}
  class PointsMaterial {}
  class GridHelper {}
  class AxesHelper {}
  class Mesh {}
}

declare module 'three' {
  export = THREE;
}

declare module 'three/examples/jsm/controls/OrbitControls.js' {
  export class OrbitControls {
    constructor(...args: any[]);
  }
}
