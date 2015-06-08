//
//  PanoramaViewer.swift
//  PanoramaStitching
//
//  Created by Paul on 08/06/2015.
//  Copyright (c) 2015 Fluid Pixel. All rights reserved.
//

import Foundation
import Cocoa
import SceneKit

class PanoramaViewer: NSViewController {
    
    @IBOutlet var scene:SCNView!
    
    var cameraNode:SCNNode!
    var ambientLightNode:SCNNode!
    var cyclinder:SCNNode!
    
    var imagePath:String? = nil {
        didSet {
            if let newPath = self.imagePath {
                if let image = NSImage(contentsOfFile: newPath) {
                    self.cyclinder.geometry?.firstMaterial?.diffuse.contents = image
                    return
                }
            }
            self.cyclinder.geometry?.firstMaterial?.diffuse.contents = NSColor.blueColor()
        }
    }
    
    override func viewDidLoad() {
        let sc = SCNScene()
        
        sc.physicsWorld.gravity = SCNVector3(x: 0.0, y: 0.0, z: 0.0)
        
        self.cameraNode = SCNNode()
        self.cameraNode.camera = SCNCamera()
        self.cameraNode.camera?.xFov = 60.0
        self.cameraNode.camera?.yFov = 60.0
        //self.cameraNode.position = SCNVector3(x: 0, y: 1.0, z: 7.0)
        self.cameraNode.position = SCNVector3(x: 0, y: 0.0, z: 0.0)
        sc.rootNode.addChildNode(self.cameraNode)

        
        self.ambientLightNode = SCNNode()
        self.ambientLightNode.light = SCNLight()
        self.ambientLightNode.light!.type = SCNLightTypeAmbient
        self.ambientLightNode.light!.color = NSColor.whiteColor()
        sc.rootNode.addChildNode(self.ambientLightNode)
        
        self.cyclinder = SCNNode(geometry: SCNSphere(radius: 3.0))
        self.cyclinder.geometry?.firstMaterial?.diffuse.contents = NSColor.blueColor()
        self.cyclinder.geometry?.firstMaterial?.cullMode = .Front
        sc.rootNode.addChildNode(self.cyclinder)
        self.cyclinder.physicsBody = SCNPhysicsBody(type: .Dynamic, shape: nil)
        self.cyclinder.physicsBody?.angularDamping = 0.8
        //self.cyclinder.physicsBody?.angularVelocity = SCNVector4(x: 0.0, y: 1.0, z: 0.0, w: 1.0)
        
        //self.scene.delegate = self
        self.scene.scene = sc
        self.scene.allowsCameraControl = false
        self.scene.showsStatistics = false
        self.scene.backgroundColor = NSColor.blackColor()
        
        self.scene.gestureRecognizers.append(NSPanGestureRecognizer(target: self, action: Selector("panGesture:")))
        self.scene.gestureRecognizers.append(NSClickGestureRecognizer(target: self, action: Selector("clickGesture:")))

    }
    func panGesture(panGesture:NSPanGestureRecognizer) {
        let multiplier = -0.00002 * CGFloat(self.cameraNode.camera!.xFov)

        let velocity = panGesture.velocityInView(self.view)
        if abs(velocity.x) > abs(velocity.y) {
            self.cyclinder.physicsBody?.applyTorque(SCNVector4(x: 0.0, y: 1.0, z: 0.0, w: velocity.x * multiplier), impulse: true)
        }
        else {
            if velocity.y > 100.0 && self.cameraNode.camera!.xFov < 120.0 {
                self.cameraNode.camera!.xFov++
                self.cameraNode.camera!.yFov = self.cameraNode.camera!.xFov
            }
            else if velocity.y < -100.0 && self.cameraNode.camera!.xFov > 1.0 {
                self.cameraNode.camera!.xFov--
                self.cameraNode.camera!.yFov = self.cameraNode.camera!.xFov
            }
        }
    }
    
    func clickGesture(clickGesture:NSClickGestureRecognizer) {
        self.cyclinder.physicsBody?.clearAllForces()
        self.cyclinder.physicsBody?.angularVelocity = SCNVector4(x: 0.0, y: 1.0, z: 0.0, w: 0.0)
    }
    

    @IBAction func zoomInButtonPressed(sender:NSButton!) {
        //SCNTransaction.begin()
        self.cameraNode.camera!.xFov++ // atan(tan(self.cameraNode.camera!.xFov * M_PI / 180.0) * 0.5) * 180.0 / M_PI
        //SCNTransaction.commit()
        println("Zoom In")
    }
    @IBAction func zoomOutButtonPressed(sender:NSButton!) {
        //SCNTransaction.begin()
        self.cameraNode.camera!.xFov-- //= atan(tan(self.cameraNode.camera!.xFov * M_PI / 180.0) * 2.0) * 180.0 / M_PI
        //SCNTransaction.commit()
        println("Zoom Out")
    }
    
}