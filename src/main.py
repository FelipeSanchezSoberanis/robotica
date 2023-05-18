from numeric_operations import AngleMode, DHParameters


def main() -> None:
    dh_parameters = DHParameters(AngleMode.DEG)

    dh_parameters.add_parameters(0, 0, 90, 60)
    dh_parameters.add_parameters(0, 0, 90, 120)
    dh_parameters.add_parameters(0, 0, 0, 135)
    dh_parameters.add_parameters(0, 4, 90, 0)
    dh_parameters.add_parameters(0, 0, 90, -60)
    dh_parameters.add_parameters(0, 1, 0, 45)

    transformation_matrices = dh_parameters.get_transformation_matrices()

    for i, matrix in enumerate(transformation_matrices):
        print(f"Matrix { i+ 1 }")
        print(matrix.round(4))

    print("Final transformation matrix")
    print(dh_parameters.get_final_transformation_matrix().round(4))


if __name__ == "__main__":
    main()
