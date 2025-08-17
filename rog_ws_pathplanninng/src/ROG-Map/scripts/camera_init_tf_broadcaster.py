#!/usr/bin/env python
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

def publish_transform():
    rospy.init_node('camera_init_to_world_tf_broadcaster')
    tf_broadcaster = tf2_ros.TransformBroadcaster()
    rate = rospy.Rate(10)  # 发布频率 10Hz

    while not rospy.is_shutdown():
        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = rospy.Time.now()
        transform_stamped.header.frame_id = "world"  # 父坐标系（目标坐标系）
        transform_stamped.child_frame_id = "camera_init"  # 子坐标系（你的数据坐标系）

        # 假设传感器坐标系与 world 坐标系原点重合，无旋转（需根据实际情况修改）
        transform_stamped.transform.translation.x = 0.0  # x 轴偏移（米）
        transform_stamped.transform.translation.y = 0.0  # y 轴偏移（米）
        transform_stamped.transform.translation.z = 0.0  # z 轴偏移（米）
        transform_stamped.transform.rotation.x = 0.0  # 四元数旋转（无旋转时为 (0,0,0,1)）
        transform_stamped.transform.rotation.y = 0.0
        transform_stamped.transform.rotation.z = 0.0
        transform_stamped.transform.rotation.w = 1.0

        tf_broadcaster.sendTransform(transform_stamped)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_transform()
    except rospy.ROSInterruptException:
        pass
