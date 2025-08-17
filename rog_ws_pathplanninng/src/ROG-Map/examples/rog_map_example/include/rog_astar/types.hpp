//
// Created by yunfan on 8/27/24.
//

#ifndef SRC_ROG_ASTAR_TYPES_HPP
#define SRC_ROG_ASTAR_TYPES_HPP

#include "rog_map/rog_map.h"
#include "memory"
#include "utils/visualization.hpp"
#define PRINT_CURRENT_LOCATION() std::cout << "Function: " << __FUNCTION__ << ", Line: " << __LINE__ << std::endl;

namespace rog_astar {
    using Vec3f = Eigen::Vector3d;
    using std::shared_ptr;
    using std::cout;
    using std::endl;
    const int ON_INF_MAP = (1 << 0);
    const int ON_PROB_MAP = (1 << 1);
    const int UNKNOWN_AS_OCCUPIED = (1 << 2);
    const int UNKNOWN_AS_FREE = (1 << 3);
    const int USE_INF_NEIGHBOR = (1 << 4);

    constexpr int DIAG = 0;
    constexpr int MANH = 1;
    constexpr int EUCL = 2;

    constexpr double inf = 1 >> 20;
    struct GridNode;
    typedef GridNode* GridNodePtr;

    struct GridNode {
        /*表示节点的状态*/
        enum enum_state {
            OPENSET = 1,
            CLOSEDSET = 2,
            UNDEFINED = 3
        } state{UNDEFINED};

        int rounds{0};
        rog_map::Vec3i id_g;
        double total_score{inf}, distance_score{inf};
        double distance_to_goal{inf};
        GridNodePtr father_ptr{nullptr};
    };

    /*比较器用于比较两个 GridNodePtr 指针所指向的节点的总得分，返回 true 表示 node1 的总得分大于 node2 的总得分。通常用于优先队列中，按照总得分从小到大排序*/
    class NodeComparator {
    public:
        bool operator()(GridNodePtr node1, GridNodePtr node2) {
            return node1->total_score > node2->total_score;
        }
    };

    /*比较两个 GridNodePtr 指针所指向的节点到目标节点的距离，返回 true 表示 node1 到目标节点的距离大于 node2 到目标节点的距离*/
    class FrontierComparator {
    public:
        bool operator()(GridNodePtr node1, GridNodePtr node2) {
            return node1->distance_to_goal > node2->distance_to_goal;
        }
    };
}

#endif //SRC_ROG_ASTAR_HPP
